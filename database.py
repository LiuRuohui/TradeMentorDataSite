import sqlite3
import os
import hashlib
import secrets
from datetime import datetime
from typing import List, Dict, Optional

class ForumDatabase:
    def __init__(self, db_path: str = "forum.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
        return conn
    
    def init_database(self):
        """初始化数据库表结构"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 创建用户表 - 扩展版本
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                user_type TEXT NOT NULL DEFAULT 'Stock Newbie',
                avatar TEXT,
                join_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                post_count INTEGER DEFAULT 0,
                reputation INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                last_login TIMESTAMP
            )
        ''')
        
        # 创建用户会话表
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
        
        # 创建分类表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                post_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建帖子表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                author_id INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                views INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (author_id) REFERENCES users (id),
                FOREIGN KEY (category_id) REFERENCES categories (id)
            )
        ''')
        
        # 创建标签表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                post_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建帖子标签关联表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS post_tags (
                post_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                PRIMARY KEY (post_id, tag_id),
                FOREIGN KEY (post_id) REFERENCES posts (id),
                FOREIGN KEY (tag_id) REFERENCES tags (id)
            )
        ''')
        
        # 创建回复表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS replies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER NOT NULL,
                author_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                likes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (post_id) REFERENCES posts (id),
                FOREIGN KEY (author_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # 插入初始数据
        self.insert_initial_data()
    
    def hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        return self.hash_password(password) == password_hash
    
    def generate_session_token(self) -> str:
        """生成会话令牌"""
        return secrets.token_urlsafe(32)
    
    def register_user(self, username: str, email: str, password: str, user_type: str) -> Dict:
        """注册新用户"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # 检查用户名是否已存在
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            if cursor.fetchone():
                raise ValueError("用户名已存在")
            
            # 检查邮箱是否已存在
            if email:
                cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
                if cursor.fetchone():
                    raise ValueError("邮箱已被注册")
            
            # 密码哈希
            password_hash = self.hash_password(password)
            
            # 插入新用户
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, user_type)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, user_type))
            
            user_id = cursor.lastrowid
            
            # 获取用户信息
            cursor.execute('''
                SELECT id, username, email, user_type, avatar, join_date
                FROM users WHERE id = ?
            ''', (user_id,))
            
            user_data = dict(cursor.fetchone())
            conn.commit()
            
            return {"success": True, "user": user_data}
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def login_user(self, username: str, password: str) -> Dict:
        """用户登录"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # 查找用户
            cursor.execute('''
                SELECT id, username, email, password_hash, user_type, avatar, join_date
                FROM users WHERE username = ? AND is_active = 1
            ''', (username,))
            
            user = cursor.fetchone()
            if not user:
                raise ValueError("用户名不存在或账户已被禁用")
            
            user_data = dict(user)
            
            # 验证密码
            if not self.verify_password(password, user_data['password_hash']):
                raise ValueError("密码错误")
            
            # 生成会话令牌
            session_token = self.generate_session_token()
            
            # 设置过期时间（7天）
            from datetime import datetime, timedelta
            expires_at = datetime.now() + timedelta(days=7)
            
            # 保存会话
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_data['id'], session_token, expires_at))
            
            # 更新最后登录时间
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_data['id'],))
            
            conn.commit()
            
            # 移除敏感信息
            del user_data['password_hash']
            
            return {
                "success": True,
                "user": user_data,
                "session_token": session_token
            }
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def verify_session(self, session_token: str) -> Optional[Dict]:
        """验证会话令牌"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # 查找有效会话
            cursor.execute('''
                SELECT u.id, u.username, u.email, u.user_type, u.avatar, u.join_date
                FROM users u
                JOIN user_sessions s ON u.id = s.user_id
                WHERE s.session_token = ? AND s.expires_at > CURRENT_TIMESTAMP
                AND u.is_active = 1
            ''', (session_token,))
            
            user = cursor.fetchone()
            if user:
                return dict(user)
            return None
            
        finally:
            conn.close()
    
    def logout_user(self, session_token: str) -> bool:
        """用户登出"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM user_sessions WHERE session_token = ?', (session_token,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """根据ID获取用户信息"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, username, email, user_type, avatar, join_date, post_count, reputation
                FROM users WHERE id = ? AND is_active = 1
            ''', (user_id,))
            
            user = cursor.fetchone()
            return dict(user) if user else None
        finally:
            conn.close()
    
    def get_user_types(self) -> List[str]:
        """获取可用的用户类型"""
        return [
            "Trading Expert",      # 交易专家
            "Stock Newbie",        # 股票新手
            "Senior Investor",     # 资深投资者
            "Market Analyst",      # 市场分析师
            "Risk Manager"         # 风险管理师
        ]
    
    def insert_initial_data(self):
        """插入初始数据"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 插入默认用户 - 添加密码和用户类型
        default_users = [
            ('Trading Expert', 'expert@example.com', '👨‍💼', 'Trading Expert', 'password123'),
            ('Stock Newbie', 'newbie@example.com', '👶', 'Stock Newbie', 'password123'),
            ('Senior Investor', 'investor@example.com', '👴', 'Senior Investor', 'password123'),
            ('Market Analyst', 'analyst@example.com', '📊', 'Market Analyst', 'password123'),
            ('Risk Manager', 'risk@example.com', '🛡️', 'Risk Manager', 'password123')
        ]
        
        for username, email, avatar, user_type, password in default_users:
            password_hash = self.hash_password(password)
            cursor.execute('''
                INSERT OR IGNORE INTO users (username, email, avatar, user_type, password_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, avatar, user_type, password_hash))
        
        # 插入默认分类
        categories = [
            ('Technical Analysis', 'Technical analysis and chart patterns'),
            ('Fundamental Analysis', 'Fundamental analysis and financial statements'),
            ('Investment Strategy', 'Investment strategies and portfolio management'),
            ('Market News', 'Market news and updates'),
            ('Beginner\'s Guide', 'Beginner-friendly guides and tutorials'),
            ('Experience Sharing', 'Personal investment experiences and stories')
        ]
        
        for name, description in categories:
            cursor.execute('''
                INSERT OR IGNORE INTO categories (name, description)
                VALUES (?, ?)
            ''', (name, description))
        
        # 插入默认标签
        tags = [
            'MACD', 'RSI', 'Bollinger Bands', 'Technical Analysis', 'Fundamentals',
            'P/E Ratio', 'Financial Analysis', 'Beginner', 'Value Investing',
            'Stock Selection', 'Investment Strategy', 'Risk Management',
            'Stop Loss', 'Trading', 'Sector Analysis', 'Market Trends'
        ]
        
        for tag_name in tags:
            cursor.execute('''
                INSERT OR IGNORE INTO tags (name)
                VALUES (?)
            ''', (tag_name,))
        
        # 插入示例帖子
        sample_posts = [
            {
                'title': 'How to Analyze Technical Indicators: MACD, RSI, and Bollinger Bands',
                'content': 'Technical analysis is a powerful tool for stock trading. In this post, I\'ll explain how to use MACD, RSI, and Bollinger Bands effectively.\n\nMACD (Moving Average Convergence Divergence) helps identify trend changes and momentum.\nRSI (Relative Strength Index) measures overbought and oversold conditions.\nBollinger Bands show price volatility and potential reversal points.',
                'author': 'Trading Expert',
                'category': 'Technical Analysis',
                'tags': ['Technical Analysis', 'MACD', 'RSI', 'Bollinger Bands'],
                'views': 156,
                'likes': 23
            },
            {
                'title': 'Beginner\'s Guide: Understanding P/E Ratio and Financial Statements',
                'content': 'As a beginner investor, understanding financial ratios is crucial. The P/E ratio is one of the most important metrics to evaluate a stock.\n\nP/E Ratio = Price per Share / Earnings per Share\n\nA lower P/E ratio might indicate undervaluation, while a higher P/E might suggest overvaluation.',
                'author': 'Stock Newbie',
                'category': 'Fundamental Analysis',
                'tags': ['Fundamentals', 'P/E Ratio', 'Financial Analysis', 'Beginner'],
                'views': 98,
                'likes': 15
            },
            {
                'title': 'My Value Investing Strategy: Finding Undervalued Stocks',
                'content': 'Value investing has been my approach for over 10 years. Here\'s my strategy for finding undervalued stocks:\n\n1. Look for companies with strong fundamentals\n2. Check for low P/E ratios compared to industry average\n3. Analyze debt levels and cash flow\n4. Consider the company\'s competitive advantage',
                'author': 'Senior Investor',
                'category': 'Investment Strategy',
                'tags': ['Value Investing', 'Stock Selection', 'Investment Strategy'],
                'views': 245,
                'likes': 31
            }
        ]
        
        for post_data in sample_posts:
            # 获取作者ID
            cursor.execute('SELECT id FROM users WHERE username = ?', (post_data['author'],))
            author_id = cursor.fetchone()['id']
            
            # 获取分类ID
            cursor.execute('SELECT id FROM categories WHERE name = ?', (post_data['category'],))
            category_id = cursor.fetchone()['id']
            
            # 插入帖子
            cursor.execute('''
                INSERT INTO posts (title, content, author_id, category_id, views, likes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (post_data['title'], post_data['content'], author_id, category_id, 
                  post_data['views'], post_data['likes']))
            
            post_id = cursor.lastrowid
            
            # 插入标签关联
            for tag_name in post_data['tags']:
                cursor.execute('SELECT id FROM tags WHERE name = ?', (tag_name,))
                tag_result = cursor.fetchone()
                if tag_result:
                    tag_id = tag_result['id']
                    cursor.execute('''
                        INSERT OR IGNORE INTO post_tags (post_id, tag_id)
                        VALUES (?, ?)
                    ''', (post_id, tag_id))
        
        conn.commit()
        conn.close()
    
    def get_all_posts(self) -> List[Dict]:
        """获取所有帖子"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                p.id, p.title, p.content, p.views, p.likes, p.created_at,
                u.username as author, u.avatar,
                c.name as category,
                GROUP_CONCAT(t.name) as tags
            FROM posts p
            JOIN users u ON p.author_id = u.id
            JOIN categories c ON p.category_id = c.id
            LEFT JOIN post_tags pt ON p.id = pt.post_id
            LEFT JOIN tags t ON pt.tag_id = t.id
            GROUP BY p.id
            ORDER BY p.created_at DESC
        ''')
        
        posts = []
        for row in cursor.fetchall():
            post = dict(row)
            post['tags'] = post['tags'].split(',') if post['tags'] else []
            posts.append(post)
        
        conn.close()
        return posts
    
    def get_categories(self) -> List[Dict]:
        """获取所有分类"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, description, post_count
            FROM categories
            ORDER BY post_count DESC
        ''')
        
        categories = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return categories
    
    def get_tags(self) -> List[Dict]:
        """获取所有标签"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, post_count
            FROM tags
            ORDER BY post_count DESC
            LIMIT 20
        ''')
        
        tags = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return tags
    
    def create_post(self, title: str, content: str, author: str, category: str, tags: List[str]) -> int:
        """创建新帖子（兼容旧版本）"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # 获取或创建作者
            cursor.execute('SELECT id FROM users WHERE username = ?', (author,))
            user_result = cursor.fetchone()
            if user_result:
                author_id = user_result['id']
            else:
                cursor.execute('INSERT INTO users (username) VALUES (?)', (author,))
                author_id = cursor.lastrowid
            
            # 获取分类ID
            cursor.execute('SELECT id FROM categories WHERE name = ?', (category,))
            category_result = cursor.fetchone()
            if not category_result:
                raise ValueError(f"Category '{category}' not found")
            category_id = category_result['id']
            
            # 插入帖子
            cursor.execute('''
                INSERT INTO posts (title, content, author_id, category_id)
                VALUES (?, ?, ?, ?)
            ''', (title, content, author_id, category_id))
            
            post_id = cursor.lastrowid
            
            # 处理标签
            for tag_name in tags:
                # 获取或创建标签
                cursor.execute('SELECT id FROM tags WHERE name = ?', (tag_name,))
                tag_result = cursor.fetchone()
                if tag_result:
                    tag_id = tag_result['id']
                else:
                    cursor.execute('INSERT INTO tags (name) VALUES (?)', (tag_name,))
                    tag_id = cursor.lastrowid
                
                # 关联帖子和标签
                cursor.execute('''
                    INSERT OR IGNORE INTO post_tags (post_id, tag_id)
                    VALUES (?, ?)
                ''', (post_id, tag_id))
            
            # 更新分类帖子计数
            cursor.execute('''
                UPDATE categories 
                SET post_count = (
                    SELECT COUNT(*) FROM posts WHERE category_id = ?
                )
                WHERE id = ?
            ''', (category_id, category_id))
            
            # 更新标签帖子计数
            for tag_name in tags:
                cursor.execute('''
                    UPDATE tags 
                    SET post_count = (
                        SELECT COUNT(*) FROM post_tags pt 
                        JOIN posts p ON pt.post_id = p.id 
                        WHERE pt.tag_id = tags.id
                    )
                    WHERE name = ?
                ''', (tag_name,))
            
            conn.commit()
            return post_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def create_post_by_user_id(self, title: str, content: str, user_id: int, category: str, tags: List[str]) -> int:
        """根据用户ID创建新帖子"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # 验证用户是否存在
            cursor.execute('SELECT id FROM users WHERE id = ? AND is_active = 1', (user_id,))
            if not cursor.fetchone():
                raise ValueError(f"User with ID {user_id} not found or inactive")
            
            # 获取分类ID
            cursor.execute('SELECT id FROM categories WHERE name = ?', (category,))
            category_result = cursor.fetchone()
            if not category_result:
                raise ValueError(f"Category '{category}' not found")
            category_id = category_result['id']
            
            # 插入帖子
            cursor.execute('''
                INSERT INTO posts (title, content, author_id, category_id)
                VALUES (?, ?, ?, ?)
            ''', (title, content, user_id, category_id))
            
            post_id = cursor.lastrowid
            
            # 处理标签
            for tag_name in tags:
                # 获取或创建标签
                cursor.execute('SELECT id FROM tags WHERE name = ?', (tag_name,))
                tag_result = cursor.fetchone()
                if tag_result:
                    tag_id = tag_result['id']
                else:
                    cursor.execute('INSERT INTO tags (name) VALUES (?)', (tag_name,))
                    tag_id = cursor.lastrowid
                
                # 关联帖子和标签
                cursor.execute('''
                    INSERT OR IGNORE INTO post_tags (post_id, tag_id)
                    VALUES (?, ?)
                ''', (post_id, tag_id))
            
            # 更新用户帖子计数
            cursor.execute('''
                UPDATE users 
                SET post_count = (
                    SELECT COUNT(*) FROM posts WHERE author_id = ?
                )
                WHERE id = ?
            ''', (user_id, user_id))
            
            # 更新分类帖子计数
            cursor.execute('''
                UPDATE categories 
                SET post_count = (
                    SELECT COUNT(*) FROM posts WHERE category_id = ?
                )
                WHERE id = ?
            ''', (category_id, category_id))
            
            # 更新标签帖子计数
            for tag_name in tags:
                cursor.execute('''
                    UPDATE tags 
                    SET post_count = (
                        SELECT COUNT(*) FROM post_tags pt 
                        JOIN posts p ON pt.post_id = p.id 
                        WHERE pt.tag_id = tags.id
                    )
                    WHERE name = ?
                ''', (tag_name,))
            
            conn.commit()
            return post_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_forum_stats(self) -> Dict:
        """获取论坛统计信息"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as total_posts FROM posts')
        total_posts = cursor.fetchone()['total_posts']
        
        cursor.execute('SELECT COUNT(*) as total_users FROM users')
        total_users = cursor.fetchone()['total_users']
        
        cursor.execute('''
            SELECT COUNT(*) as posts_today 
            FROM posts 
            WHERE DATE(created_at) = DATE('now')
        ''')
        posts_today = cursor.fetchone()['posts_today']
        
        cursor.execute('SELECT COUNT(*) as total_categories FROM categories')
        total_categories = cursor.fetchone()['total_categories']
        
        conn.close()
        
        return {
            'total_posts': total_posts,
            'total_users': total_users,
            'posts_today': posts_today,
            'total_categories': total_categories
        }
    
    def get_user_posts(self, user_id: int) -> List[Dict]:
        """获取指定用户的帖子"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                p.id, p.title, p.content, p.views, p.likes, p.created_at,
                u.username as author, u.avatar, u.user_type,
                c.name as category,
                GROUP_CONCAT(t.name) as tags
            FROM posts p
            JOIN users u ON p.author_id = u.id
            JOIN categories c ON p.category_id = c.id
            LEFT JOIN post_tags pt ON p.id = pt.post_id
            LEFT JOIN tags t ON pt.tag_id = t.id
            WHERE p.author_id = ?
            GROUP BY p.id
            ORDER BY p.created_at DESC
        ''', (user_id,))
        
        posts = []
        for row in cursor.fetchall():
            post = dict(row)
            post['tags'] = post['tags'].split(',') if post['tags'] else []
            posts.append(post)
        
        conn.close()
        return posts

    def get_post_by_id(self, post_id: int) -> dict:
        """根据帖子ID获取帖子详情"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                p.id, p.title, p.content, p.views, p.likes, p.created_at,
                u.username as author, u.avatar,
                c.name as category,
                GROUP_CONCAT(t.name) as tags
            FROM posts p
            JOIN users u ON p.author_id = u.id
            JOIN categories c ON p.category_id = c.id
            LEFT JOIN post_tags pt ON p.id = pt.post_id
            LEFT JOIN tags t ON pt.tag_id = t.id
            WHERE p.id = ?
            GROUP BY p.id
        ''', (post_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            post = dict(row)
            post['tags'] = post['tags'].split(',') if post['tags'] else []
            return post
        return None

    def get_replies_by_post_id(self, post_id: int) -> list:
        """获取某个帖子的所有评论（按时间升序）"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT r.id, r.content, r.likes, r.created_at, u.username as author
            FROM replies r
            JOIN users u ON r.author_id = u.id
            WHERE r.post_id = ?
            ORDER BY r.created_at ASC
        ''', (post_id,))
        replies = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return replies

    def add_reply(self, post_id: int, author_id: int, content: str) -> int:
        """添加评论"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO replies (post_id, author_id, content)
            VALUES (?, ?, ?)
        ''', (post_id, author_id, content))
        reply_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return reply_id

    def like_post(self, post_id: int) -> int:
        """给帖子点赞，返回最新点赞数"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE posts SET likes = likes + 1 WHERE id = ?', (post_id,))
        conn.commit()
        cursor.execute('SELECT likes FROM posts WHERE id = ?', (post_id,))
        likes = cursor.fetchone()[0]
        conn.close()
        return likes

# 创建全局数据库实例
db = ForumDatabase() 