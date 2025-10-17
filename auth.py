"""
Authentication Module for RenewAI
Handles user registration, login, and session management
"""
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import session, redirect, url_for, request

# Database configuration
DB_NAME = 'renewai_users.db'

def get_db_connection():
    """Create database connection"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with users table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            company TEXT,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Create sessions table for tracking active sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized successfully")

def hash_password(password, salt=None):
    """Hash password with salt using SHA-256"""
    if salt is None:
        salt = secrets.token_hex(32)
    
    # Combine password and salt, then hash
    pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return pwd_hash, salt

def verify_password(password, password_hash, salt):
    """Verify password against hash"""
    test_hash, _ = hash_password(password, salt)
    return test_hash == password_hash

def register_user(name, email, password, company=None):
    """Register a new user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if email already exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return {'success': False, 'error': 'Email already registered'}
        
        # Hash password
        pwd_hash, salt = hash_password(password)
        
        # Insert user
        cursor.execute('''
            INSERT INTO users (name, email, company, password_hash, salt)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, email, company, pwd_hash, salt))
        
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        return {
            'success': True,
            'user_id': user_id,
            'message': 'User registered successfully'
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def login_user(email, password):
    """Authenticate user and create session"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get user by email
        cursor.execute('''
            SELECT id, name, email, company, password_hash, salt
            FROM users WHERE email = ?
        ''', (email,))
        
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return {'success': False, 'error': 'Invalid email or password'}
        
        # Verify password
        if not verify_password(password, user['password_hash'], user['salt']):
            conn.close()
            return {'success': False, 'error': 'Invalid email or password'}
        
        # Update last login
        cursor.execute('''
            UPDATE users SET last_login = ? WHERE id = ?
        ''', (datetime.now(), user['id']))
        
        # Create session token
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=7)
        
        cursor.execute('''
            INSERT INTO user_sessions (user_id, session_token, expires_at)
            VALUES (?, ?, ?)
        ''', (user['id'], session_token, expires_at))
        
        conn.commit()
        conn.close()
        
        return {
            'success': True,
            'user': {
                'id': user['id'],
                'name': user['name'],
                'email': user['email'],
                'company': user['company']
            },
            'session_token': session_token
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def logout_user(session_token):
    """Logout user by removing session"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM user_sessions WHERE session_token = ?', (session_token,))
        conn.commit()
        conn.close()
        
        return {'success': True, 'message': 'Logged out successfully'}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def validate_session(session_token):
    """Validate session token and return user data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.id, u.name, u.email, u.company, s.expires_at
            FROM users u
            JOIN user_sessions s ON u.id = s.user_id
            WHERE s.session_token = ?
        ''', (session_token,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return {'valid': False, 'error': 'Invalid session'}
        
        # Check if session expired
        expires_at = datetime.fromisoformat(result['expires_at'])
        if datetime.now() > expires_at:
            logout_user(session_token)
            return {'valid': False, 'error': 'Session expired'}
        
        return {
            'valid': True,
            'user': {
                'id': result['id'],
                'name': result['name'],
                'email': result['email'],
                'company': result['company']
            }
        }
    
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def get_user_by_id(user_id):
    """Get user information by ID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, email, company, created_at, last_login
            FROM users WHERE id = ?
        ''', (user_id,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'success': True,
                'user': dict(user)
            }
        return {'success': False, 'error': 'User not found'}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Decorator for protected routes
def login_required(f):
    """Decorator to protect routes that require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_token = session.get('session_token')
        
        if not session_token:
            return redirect(url_for('landing'))
        
        validation = validate_session(session_token)
        if not validation['valid']:
            session.clear()
            return redirect(url_for('landing'))
        
        # Add user info to request context
        request.current_user = validation['user']
        return f(*args, **kwargs)
    
    return decorated_function

# Initialize database on import
init_db()