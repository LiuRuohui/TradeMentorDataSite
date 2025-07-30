import sqlite3
import os
import hashlib
import secrets
from datetime import datetime
from typing import List, Dict, Optional
import json

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
        
        # 创建考试表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                total_questions INTEGER NOT NULL,
                passing_score INTEGER NOT NULL,
                time_limit INTEGER DEFAULT 30,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # 创建考试题目表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exam_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_id INTEGER NOT NULL,
                question_text TEXT NOT NULL,
                option_a TEXT NOT NULL,
                option_b TEXT NOT NULL,
                option_c TEXT NOT NULL,
                option_d TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                score INTEGER DEFAULT 1,
                question_order INTEGER NOT NULL,
                FOREIGN KEY (exam_id) REFERENCES exams (id)
            )
        ''')
        
        # 创建考试记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exam_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                exam_id INTEGER NOT NULL,
                score INTEGER NOT NULL,
                total_score INTEGER NOT NULL,
                percentage INTEGER NOT NULL,
                passed BOOLEAN NOT NULL,
                answers TEXT,  -- JSON格式存储答案
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (exam_id) REFERENCES exams (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # 插入初始数据
        self.insert_initial_data()
        # 插入示例考试数据
        self.insert_sample_exams()
    
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
                'likes': 23,
                'created_at': '2025-01-15 10:30:00'
            },
            {
                'title': 'Beginner\'s Guide: Understanding P/E Ratio and Financial Statements',
                'content': 'As a beginner investor, understanding financial ratios is crucial. The P/E ratio is one of the most important metrics to evaluate a stock.\n\nP/E Ratio = Price per Share / Earnings per Share\n\nA lower P/E ratio might indicate undervaluation, while a higher P/E might suggest overvaluation.',
                'author': 'Stock Newbie',
                'category': 'Fundamental Analysis',
                'tags': ['Fundamentals', 'P/E Ratio', 'Financial Analysis', 'Beginner'],
                'views': 98,
                'likes': 15,
                'created_at': '2025-01-16 14:20:00'
            },
            {
                'title': 'My Value Investing Strategy: Finding Undervalued Stocks',
                'content': 'Value investing has been my approach for over 10 years. Here\'s my strategy for finding undervalued stocks:\n\n1. Look for companies with strong fundamentals\n2. Check for low P/E ratios compared to industry average\n3. Analyze debt levels and cash flow\n4. Consider the company\'s competitive advantage',
                'author': 'Senior Investor',
                'category': 'Investment Strategy',
                'tags': ['Value Investing', 'Stock Selection', 'Investment Strategy'],
                'views': 245,
                'likes': 31,
                'created_at': '2025-01-17 09:15:00'
            }
        ]
        
        for post_data in sample_posts:
            # 获取作者ID
            cursor.execute('SELECT id FROM users WHERE username = ?', (post_data['author'],))
            result = cursor.fetchone()
            if result:
                author_id = result['id']
            else:
                return  # 查不到作者直接跳出
            # 获取分类ID
            cursor.execute('SELECT id FROM categories WHERE name = ?', (post_data['category'],))
            category_result = cursor.fetchone()
            if category_result:
                category_id = category_result['id']
            else:
                return  # 查不到分类直接跳出
            # 插入帖子
            cursor.execute('''
                INSERT INTO posts (title, content, author_id, category_id, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (post_data['title'], post_data['content'], author_id, category_id, post_data['created_at']))
            
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
            # 查询评论数
            cursor2 = conn.cursor()
            cursor2.execute('SELECT COUNT(*) FROM replies WHERE post_id = ?', (post['id'],))
            post['replies_count'] = cursor2.fetchone()[0]
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
        """点赞帖子"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('UPDATE posts SET likes = likes + 1 WHERE id = ?', (post_id,))
            conn.commit()
            
            cursor.execute('SELECT likes FROM posts WHERE id = ?', (post_id,))
            result = cursor.fetchone()
            return result['likes'] if result else 0
        except Exception as e:
            print(f"Error liking post: {e}")
            return 0
        finally:
            conn.close()
    
    # ================= 考试相关方法 =================
    
    def get_all_exams(self) -> List[Dict]:
        """获取所有考试列表"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, title, description, total_questions, passing_score, time_limit, created_at
                FROM exams 
                WHERE is_active = 1 
                ORDER BY created_at DESC
            ''')
            exams = []
            for row in cursor.fetchall():
                exams.append(dict(row))
            return exams
        except Exception as e:
            print(f"Error getting exams: {e}")
            return []
        finally:
            conn.close()
    
    def get_exam_by_id(self, exam_id: int) -> Optional[Dict]:
        """根据ID获取考试详情"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, title, description, total_questions, passing_score, time_limit, created_at
                FROM exams 
                WHERE id = ? AND is_active = 1
            ''', (exam_id,))
            result = cursor.fetchone()
            return dict(result) if result else None
        except Exception as e:
            print(f"Error getting exam: {e}")
            return None
        finally:
            conn.close()
    
    def get_exam_questions(self, exam_id: int) -> List[Dict]:
        """获取考试题目"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, question_text, option_a, option_b, option_c, option_d, correct_answer, score, question_order
                FROM exam_questions 
                WHERE exam_id = ? 
                ORDER BY question_order
            ''', (exam_id,))
            questions = []
            for row in cursor.fetchall():
                questions.append(dict(row))
            return questions
        except Exception as e:
            print(f"Error getting exam questions: {e}")
            return []
        finally:
            conn.close()
    
    def save_exam_record(self, user_id: int, exam_id: int, score: int, total_score: int, 
                        percentage: int, passed: bool, answers: str) -> int:
        """保存考试记录"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO exam_records (user_id, exam_id, score, total_score, percentage, passed, answers, end_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, exam_id, score, total_score, percentage, passed, answers))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error saving exam record: {e}")
            return 0
        finally:
            conn.close()
    
    def get_user_exam_records(self, user_id: int) -> List[Dict]:
        """获取用户的考试记录"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT er.id, er.score, er.total_score, er.percentage, er.passed, er.start_time, er.end_time,
                       e.title, e.description
                FROM exam_records er
                JOIN exams e ON er.exam_id = e.id
                WHERE er.user_id = ?
                ORDER BY er.end_time DESC
            ''', (user_id,))
            records = []
            for row in cursor.fetchall():
                records.append(dict(row))
            return records
        except Exception as e:
            print(f"Error getting user exam records: {e}")
            return []
        finally:
            conn.close()
    
    def get_exam_record_detail(self, record_id: int) -> Optional[Dict]:
        """获取考试记录详情"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT er.id, er.score, er.total_score, er.percentage, er.passed, er.start_time, er.end_time,
                       er.answers, e.title, e.description, e.total_questions, e.passing_score
                FROM exam_records er
                JOIN exams e ON er.exam_id = e.id
                WHERE er.id = ?
            ''', (record_id,))
            result = cursor.fetchone()
            if result:
                record = dict(result)
                # 解析答案JSON
                if record['answers']:
                    record['answers'] = json.loads(record['answers'])
                return record
            return None
        except Exception as e:
            print(f"Error getting exam record detail: {e}")
            return None
        finally:
            conn.close()
    
    def insert_sample_exams(self):
        """插入示例考试数据"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # 检查是否已有考试数据
            cursor.execute('SELECT COUNT(*) as count FROM exams')
            if cursor.fetchone()['count'] > 0:
                return  # 已有数据，不重复插入
            
            # 插入考试
            exams_data = [
                {
                    'title': 'Stock Investment Basic Knowledge Test',
                    'description': 'Test your mastery of basic stock investment knowledge',
                    'total_questions': 5,
                    'passing_score': 60,
                    'time_limit': 30
                },
                {
                    'title': 'Technical Analysis Advanced Test',
                    'description': 'Test your understanding of technical analysis indicators and methods',
                    'total_questions': 5,
                    'passing_score': 60,
                    'time_limit': 30
                },
                {
                    'title': 'Fundamental Analysis Test',
                    'description': 'Test your ability in company financial analysis and fundamental analysis',
                    'total_questions': 5,
                    'passing_score': 60,
                    'time_limit': 30
                },
                {
                    'title': 'Risk Management Test',
                    'description': 'Test your understanding of investment risk management and asset allocation',
                    'total_questions': 5,
                    'passing_score': 60,
                    'time_limit': 30
                },
                {
                    'title': 'Market Psychology Test',
                    'description': 'Test your understanding of market psychology and investment psychology',
                    'total_questions': 5,
                    'passing_score': 60,
                    'time_limit': 30
                }
            ]
            
            for exam_data in exams_data:
                cursor.execute('''
                    INSERT INTO exams (title, description, total_questions, passing_score, time_limit)
                    VALUES (?, ?, ?, ?, ?)
                ''', (exam_data['title'], exam_data['description'], exam_data['total_questions'], 
                     exam_data['passing_score'], exam_data['time_limit']))
                exam_id = cursor.lastrowid
                
                # 插入对应的题目
                questions_data = self._get_exam_questions_by_type(exam_data['title'])
                for i, question in enumerate(questions_data, 1):
                    cursor.execute('''
                        INSERT INTO exam_questions (exam_id, question_text, option_a, option_b, option_c, option_d, correct_answer, score, question_order)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (exam_id, question['question'], question['option_a'], question['option_b'], 
                         question['option_c'], question['option_d'], question['correct'], 1, i))
            
            conn.commit()
            print("Sample exam data inserted successfully")
        except Exception as e:
            print(f"Error inserting sample exams: {e}")
        finally:
            conn.close()
    
    def _get_exam_questions_by_type(self, exam_title: str) -> List[Dict]:
        """根据考试类型获取题目"""
        if 'Basic Knowledge' in exam_title:
            return [
                {
                    'question': 'What aspects does fundamental analysis of stocks primarily focus on?',
                    'option_a': 'Company financial status',
                    'option_b': 'Technical indicators',
                    'option_c': 'Market sentiment',
                    'option_d': 'All of the above',
                    'correct': 'A'
                },
                {
                    'question': 'Price-to-earnings ratio (PE) is an important indicator for measuring stock valuation. What is its calculation formula?',
                    'option_a': 'Share price / Earnings per share',
                    'option_b': 'Share price / Net asset value per share',
                    'option_c': 'Share price / Cash flow per share',
                    'option_d': 'Share price / Dividend per share',
                    'correct': 'A'
                },
                {
                    'question': 'Which of the following is NOT a common technical indicator?',
                    'option_a': 'Moving Average',
                    'option_b': 'Relative Strength Index (RSI)',
                    'option_c': 'Company Revenue',
                    'option_d': 'MACD',
                    'correct': 'C'
                },
                {
                    'question': 'What does a stock\'s beta measure?',
                    'option_a': 'The stock\'s volatility compared to the market',
                    'option_b': 'The stock\'s dividend yield',
                    'option_c': 'The stock\'s market capitalization',
                    'option_d': 'The stock\'s earnings per share',
                    'correct': 'A'
                },
                {
                    'question': 'What is the primary purpose of diversification in investment?',
                    'option_a': 'To maximize returns',
                    'option_b': 'To reduce risk',
                    'option_c': 'To minimize taxes',
                    'option_d': 'To increase trading frequency',
                    'correct': 'B'
                }
            ]
        elif 'Technical Analysis' in exam_title:
            return [
                {
                    'question': 'In MACD indicator, when the DIF line crosses above the DEA line, what does this typically indicate?',
                    'option_a': 'Sell signal',
                    'option_b': 'Buy signal',
                    'option_c': 'Wait signal',
                    'option_d': 'No meaning',
                    'correct': 'B'
                },
                {
                    'question': 'When RSI indicator exceeds 70, what does this typically indicate?',
                    'option_a': 'Stock oversold',
                    'option_b': 'Stock overbought',
                    'option_c': 'Stock normal',
                    'option_d': 'Indicator invalid',
                    'correct': 'B'
                },
                {
                    'question': 'What does a doji pattern in candlestick charts typically indicate?',
                    'option_a': 'Strong upward signal',
                    'option_b': 'Strong downward signal',
                    'option_c': 'Market indecision',
                    'option_d': 'No special meaning',
                    'correct': 'C'
                },
                {
                    'question': 'In Bollinger Bands indicator, when stock price touches the lower band, what does this typically indicate?',
                    'option_a': 'Overbought signal',
                    'option_b': 'Oversold signal',
                    'option_c': 'Normal fluctuation',
                    'option_d': 'Trend reversal',
                    'correct': 'B'
                },
                {
                    'question': 'What is the role of volume in technical analysis?',
                    'option_a': 'Confirm price trends',
                    'option_b': 'Predict future prices',
                    'option_c': 'Calculate technical indicators',
                    'option_d': 'All of the above',
                    'correct': 'A'
                }
            ]
        elif 'Fundamental Analysis' in exam_title:
            return [
                {
                    'question': 'Which of the following financial ratios best reflects a company\'s profitability?',
                    'option_a': 'Debt-to-equity ratio',
                    'option_b': 'Return on Equity (ROE)',
                    'option_c': 'Current ratio',
                    'option_d': 'Inventory turnover ratio',
                    'correct': 'B'
                },
                {
                    'question': 'What is a company\'s free cash flow?',
                    'option_a': 'Net profit',
                    'option_b': 'Operating cash flow minus capital expenditures',
                    'option_c': 'Total assets',
                    'option_d': 'Shareholders\' equity',
                    'correct': 'B'
                },
                {
                    'question': 'Which indicator best reflects a company\'s growth potential?',
                    'option_a': 'Price-to-earnings ratio',
                    'option_b': 'Revenue growth rate',
                    'option_c': 'Dividend yield',
                    'option_d': 'Price-to-book ratio',
                    'correct': 'B'
                },
                {
                    'question': 'What category does goodwill belong to in a company\'s balance sheet?',
                    'option_a': 'Current assets',
                    'option_b': 'Intangible assets',
                    'option_c': 'Long-term liabilities',
                    'option_d': 'Shareholders\' equity',
                    'correct': 'B'
                },
                {
                    'question': 'Which of the following is NOT an important indicator for measuring a company\'s financial health?',
                    'option_a': 'Current ratio',
                    'option_b': 'Debt-to-equity ratio',
                    'option_c': 'Stock price',
                    'option_d': 'Interest coverage ratio',
                    'correct': 'C'
                }
            ]
        elif 'Risk Management' in exam_title:
            return [
                {
                    'question': 'Which of the following investment strategies has the highest risk?',
                    'option_a': 'Index fund investment',
                    'option_b': 'Single stock investment',
                    'option_c': 'Bond investment',
                    'option_d': 'Money market fund',
                    'correct': 'B'
                },
                {
                    'question': 'What is the main purpose of a stop-loss order?',
                    'option_a': 'Lock in profits',
                    'option_b': 'Limit losses',
                    'option_c': 'Increase returns',
                    'option_d': 'Reduce trading costs',
                    'correct': 'B'
                },
                {
                    'question': 'In an investment portfolio, when the correlation between different asset classes is lower, how does risk change?',
                    'option_a': 'Risk increases',
                    'option_b': 'Risk decreases',
                    'option_c': 'Risk remains unchanged',
                    'option_d': 'Cannot determine',
                    'correct': 'B'
                },
                {
                    'question': 'Which of the following is NOT an effective risk management strategy?',
                    'option_a': 'Asset allocation',
                    'option_b': 'Regular rebalancing',
                    'option_c': 'Full position trading',
                    'option_d': 'Diversified investment',
                    'correct': 'C'
                },
                {
                    'question': 'What does VaR (Value at Risk) measure?',
                    'option_a': 'Maximum possible return',
                    'option_b': 'Maximum possible loss',
                    'option_c': 'Average return',
                    'option_d': 'Return standard deviation',
                    'correct': 'B'
                }
            ]
        else:  # Market Psychology
            return [
                {
                    'question': 'Herd behavior in investment manifests as?',
                    'option_a': 'Independent thinking',
                    'option_b': 'Following crowd behavior',
                    'option_c': 'Contrarian investment',
                    'option_d': 'Long-term holding',
                    'correct': 'B'
                },
                {
                    'question': 'Anchoring effect refers to investors being easily influenced by what?',
                    'option_a': 'Historical prices',
                    'option_b': 'Future expectations',
                    'option_c': 'Market sentiment',
                    'option_d': 'Expert opinions',
                    'correct': 'A'
                },
                {
                    'question': 'Loss aversion psychology causes investors to?',
                    'option_a': 'Sell profitable stocks too early',
                    'option_b': 'Hold losing stocks for too long',
                    'option_c': 'Make rational decisions',
                    'option_d': 'Trade frequently',
                    'correct': 'A'
                },
                {
                    'question': 'Overconfidence in investment typically manifests as?',
                    'option_a': 'Excessive trading',
                    'option_b': 'Conservative investment',
                    'option_c': 'Diversified investment',
                    'option_d': 'Long-term holding',
                    'correct': 'A'
                },
                {
                    'question': 'Which of the following is NOT a common investment psychology bias?',
                    'option_a': 'Confirmation bias',
                    'option_b': 'Anchoring effect',
                    'option_c': 'Rational decision-making',
                    'option_d': 'Herd behavior',
                    'correct': 'C'
                }
            ]

# 创建全局数据库实例
db = ForumDatabase() 