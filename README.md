# WebSocket Test Application

A FastAPI-based WebSocket application with inactivity handling, rate limiting, and authentication.

## Features

- WebSocket support with conversation management
- JWT-based authentication
- Configurable inactivity timeout
- Rate limiting based on user plans
- Simple web interface for testing
- Redis-based connection and rate limit management
- Comprehensive error handling and logging
- Secure password hashing

## Prerequisites

- Python 3.8+
- Redis server
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd websocket-test
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with the following variables:
```
REDIS_HOST=localhost
REDIS_PORT=6379
INACTIVITY_TIMEOUT=300  # in seconds
SECRET_KEY=your-secret-key-here  # Change this to a secure secret key
```

## Running the Application

1. Start Redis server:
```bash
redis-server
```

2. Start the FastAPI application:
```bash
uvicorn app.main:app --reload
```

3. Open the web interface:
- Open `static/index.html` in your web browser
- Use one of the following test accounts:
  - Username: user1, Password: password (Free plan)
  - Username: user2, Password: password (Basic plan)
  - Username: user3, Password: password (Premium plan)
- Enter a Conversation ID
- Click "Connect" to establish the WebSocket connection
- Start sending messages

## Authentication

The application uses JWT (JSON Web Tokens) for authentication. Tokens are obtained through the `/token` endpoint and are required for WebSocket connections.

## Rate Limiting Plans

The application implements different rate limits based on user plans:

- Free: 100 requests per hour
- Basic: 1,000 requests per hour
- Premium: 10,000 requests per hour

## Inactivity Handling

The server will automatically close connections that have been inactive for the configured timeout period (default: 5 minutes).

## Error Handling

The application provides comprehensive error handling:
- Authentication errors
- Rate limit exceeded
- Connection inactivity
- Invalid WebSocket connections
- Server errors
- Input validation
- Message parsing errors

## Logging

The application logs various events:
- Connection attempts
- Authentication successes/failures
- Rate limit violations
- Inactivity timeouts
- Error conditions
- Message processing

## Project Structure

```
websocket-test/
├── app/
│   └── main.py          # FastAPI application
├── static/
│   └── index.html       # Web interface
├── requirements.txt     # Python dependencies
├── .env                # Environment variables
└── README.md           # This file
```

## API Endpoints

- Authentication: `POST /token`
- WebSocket: `ws://localhost:8000/ws/{user_id}/{conversation_id}?token={token}`

## Security Features

- JWT-based authentication
- Password hashing with bcrypt
- Rate limiting
- Input validation
- Secure WebSocket connections
- Error message sanitization

## Contributing

Feel free to submit issues and enhancement requests.