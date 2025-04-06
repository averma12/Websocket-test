from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
import asyncio
import json
import logging
from typing import Dict, Set, Optional
from enum import Enum
import redis
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from passlib.context import CryptContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class TokenData(BaseModel):
    user_id: str
    plan: str

class User(BaseModel):
    username: str
    plan: str

class LoginRequest(BaseModel):
    username: str
    password: str

class MessageType(str, Enum):
    TEXT = "text"
    COMMAND = "command"
    ECHO = "echo"

class Message(BaseModel):
    type: MessageType
    content: str
    timestamp: Optional[datetime] = None

# This would normally come from a database
fake_users_db = {
    "user1": {
        "username": "user1",
        "hashed_password": pwd_context.hash("password"),
        "plan": "free"
    },
    "user2": {
        "username": "user2",
        "hashed_password": pwd_context.hash("password"),
        "plan": "basic"
    },
    "user3": {
        "username": "user3",
        "hashed_password": pwd_context.hash("password"),
        "plan": "premium"
    }
}

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection for rate limiting
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0
)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        plan: str = payload.get("plan")
        if user_id is None or plan is None:
            raise credentials_exception
        return TokenData(user_id=user_id, plan=plan)
    except JWTError:
        raise credentials_exception

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.inactivity_timeout = int(os.getenv("INACTIVITY_TIMEOUT", 300))  # 5 minutes default
        self.warning_threshold = int(os.getenv("WARNING_THRESHOLD", 240))  # 4 minutes default

    async def connect(self, websocket: WebSocket, user_id: str, conversation_id: str):
        try:
            await websocket.accept()
            if user_id not in self.active_connections:
                self.active_connections[user_id] = {}
            self.active_connections[user_id][conversation_id] = websocket
            logger.info(f"User {user_id} connected to conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Error connecting user {user_id}: {str(e)}")
            raise

    def disconnect(self, user_id: str, conversation_id: str):
        try:
            if user_id in self.active_connections:
                if conversation_id in self.active_connections[user_id]:
                    del self.active_connections[user_id][conversation_id]
                    logger.info(f"User {user_id} disconnected from conversation {conversation_id}")
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
        except Exception as e:
            logger.error(f"Error disconnecting user {user_id}: {str(e)}")

    async def check_inactivity(self):
        while True:
            try:
                current_time = datetime.now()
                for user_id, conversations in list(self.active_connections.items()):
                    for conversation_id, websocket in list(conversations.items()):
                        last_activity = redis_client.get(f"last_activity:{user_id}:{conversation_id}")
                        if last_activity:
                            last_activity_time = datetime.fromisoformat(last_activity.decode())
                            inactive_seconds = (current_time - last_activity_time).seconds
                            
                            # Send warning message if approaching timeout
                            if inactive_seconds >= self.warning_threshold and inactive_seconds < self.inactivity_timeout:
                                warning_seconds = self.inactivity_timeout - inactive_seconds
                                await self.send_personal_message(
                                    {
                                        "type": "warning",
                                        "content": f"Connection will close in {warning_seconds} seconds due to inactivity",
                                        "timestamp": datetime.now().isoformat()
                                    },
                                    user_id,
                                    conversation_id
                                )
                            
                            # Close connection if timeout reached
                            if inactive_seconds >= self.inactivity_timeout:
                                logger.info(f"Closing inactive connection for user {user_id} in conversation {conversation_id}")
                                await self.send_personal_message(
                                    {
                                        "type": "system",
                                        "content": "Connection closed due to inactivity",
                                        "timestamp": datetime.now().isoformat()
                                    },
                                    user_id,
                                    conversation_id
                                )
                                await websocket.close()
                                self.disconnect(user_id, conversation_id)
            except Exception as e:
                logger.error(f"Error in inactivity check: {str(e)}")
            await asyncio.sleep(10)  # Check every 10 seconds for more responsive warnings

    async def send_personal_message(self, message: dict, user_id: str, conversation_id: str):
        try:
            if user_id in self.active_connections and conversation_id in self.active_connections[user_id]:
                await self.active_connections[user_id][conversation_id].send_json(message)
        except Exception as e:
            logger.error(f"Error sending message to user {user_id}: {str(e)}")
            self.disconnect(user_id, conversation_id)

manager = ConnectionManager()

# Rate limiting function
def check_rate_limit(user_id: str, plan: str) -> bool:
    try:
        rate_limits = {
            "free": {"requests": 100, "window": 3600},  # 100 requests per hour
            "basic": {"requests": 1000, "window": 3600},  # 1000 requests per hour
            "premium": {"requests": 10000, "window": 3600}  # 10000 requests per hour
        }
        
        current_time = datetime.now()
        key = f"rate_limit:{user_id}"
        
        # Get current count and window start
        count, window_start = redis_client.hmget(key, ["count", "window_start"])
        
        if not count or not window_start:
            # First request
            redis_client.hmset(key, {
                "count": 1,
                "window_start": current_time.isoformat()
            })
            return True
        
        window_start = datetime.fromisoformat(window_start.decode())
        if (current_time - window_start).seconds > rate_limits[plan]["window"]:
            # Reset window
            redis_client.hmset(key, {
                "count": 1,
                "window_start": current_time.isoformat()
            })
            return True
        
        count = int(count.decode())
        if count >= rate_limits[plan]["requests"]:
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return False
        
        redis_client.hincrby(key, "count", 1)
        return True
    except Exception as e:
        logger.error(f"Error in rate limiting for user {user_id}: {str(e)}")
        return True  # Fail open in case of errors

@app.websocket("/ws/{user_id}/{conversation_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    conversation_id: str,
    token: str = None
):
    try:
        logger.info(f"WebSocket connection attempt - User: {user_id}, Conversation: {conversation_id}")
        
        if not token:
            logger.warning(f"No token provided for user {user_id}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Token required")
            return

        # Verify token and get user data
        try:
            token_data = await get_current_user(token)
            logger.info(f"Token verified for user {user_id}")
        except Exception as e:
            logger.error(f"Token verification failed for user {user_id}: {str(e)}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
            return

        if token_data.user_id != user_id:
            logger.warning(f"Token user mismatch: {token_data.user_id} != {user_id}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid user")
            return

        if not check_rate_limit(user_id, token_data.plan):
            logger.warning(f"Rate limit exceeded for user {user_id}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Rate limit exceeded")
            return

        await manager.connect(websocket, user_id, conversation_id)
        logger.info(f"WebSocket connection established - User: {user_id}, Conversation: {conversation_id}")
        
        try:
            while True:
                raw_data = await websocket.receive_json()
                logger.debug(f"Received message from user {user_id}: {raw_data}")
                
                # Update last activity timestamp
                redis_client.set(
                    f"last_activity:{user_id}:{conversation_id}",
                    datetime.now().isoformat()
                )

                # Process the message based on its type
                try:
                    message = Message(
                        type=raw_data.get("type", MessageType.TEXT),
                        content=raw_data.get("content", ""),
                        timestamp=datetime.now()
                    )

                    response = None
                    if message.type == MessageType.TEXT:
                        # Simple text response
                        response = {
                            "type": "text",
                            "content": f"Echo: {message.content}",
                            "timestamp": datetime.now().isoformat()
                        }
                    elif message.type == MessageType.COMMAND:
                        # Handle commands
                        if message.content.startswith("/"):
                            command = message.content[1:].lower()
                            if command == "help":
                                response = {
                                    "type": "system",
                                    "content": "Available commands: /help, /time, /plan",
                                    "timestamp": datetime.now().isoformat()
                                }
                            elif command == "time":
                                response = {
                                    "type": "system",
                                    "content": f"Server time: {datetime.now().isoformat()}",
                                    "timestamp": datetime.now().isoformat()
                                }
                            elif command == "plan":
                                response = {
                                    "type": "system",
                                    "content": f"Your current plan: {token_data.plan}",
                                    "timestamp": datetime.now().isoformat()
                                }
                    elif message.type == MessageType.ECHO:
                        # Echo the message back
                        response = {
                            "type": "echo",
                            "content": message.content,
                            "timestamp": datetime.now().isoformat()
                        }

                    if response:
                        await manager.send_personal_message(response, user_id, conversation_id)
                    else:
                        await manager.send_personal_message({
                            "type": "error",
                            "content": "Unknown message type or invalid command",
                            "timestamp": datetime.now().isoformat()
                        }, user_id, conversation_id)

                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await manager.send_personal_message({
                        "type": "error",
                        "content": f"Error processing message: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }, user_id, conversation_id)

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected - User: {user_id}, Conversation: {conversation_id}")
            manager.disconnect(user_id, conversation_id)
        except Exception as e:
            logger.error(f"Error in WebSocket connection for user {user_id}: {str(e)}")
            manager.disconnect(user_id, conversation_id)
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {str(e)}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(manager.check_inactivity())

@app.post("/token")
async def login(request: LoginRequest):
    user = fake_users_db.get(request.username)
    if not user or not pwd_context.verify(request.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "plan": user["plan"]},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 