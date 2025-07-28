from dataclasses import dataclass, field
from typing import List, Set, Dict
import re
import pickle
from datetime import datetime
import bcrypt
from abc import ABC, abstractmethod




@dataclass
class User:

    """Класс для представления пользователя в системе
    
    Attributes:
        id (int): уникальный идентификатор пользователя
        username (str): логин пользователя
        existing_usernames (set): множество уникальных логинов пользователей
        password (str): пароль пользователя (хэшированный)
        email (str): email пользователя
        transactions (List[Transaction]): список транзакций пользователя
        requests (List[Prediction]): список запросов пользователя
      
    """
    
    id: int
    username: str
    password: str
    email: str
    balance: Balance = field(default_factory=lambda: Balance(balance=10.0))
    transactions: List['Transaction'] = field(default_factory=list)
    predictions: List['Prediction'] = field(default_factory=list)

    __existing_usernames: Set[str] = field(default_factory=set, init=False)
    

    def __post_init__(self) -> None:
        self._validate_username()
        self._validate_email()
        self._validate_password()
        self.password = self._hash_password(self.password)
    
    def _validate_username(self) -> None:
        """Проверяет уникальность введенного логина при регистрации."""
        if self.username in self.__existing_usernames:
            raise ValueError(f"Username '{self.username}' already exists")
        # Сохраняем уникальный логин
        self.__existing_usernames.add(self.username)

    def _validate_email(self) -> None:
        """Проверяет корректность email."""
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not email_pattern.match(self.email):
            raise ValueError("Invalid email format")

    def _validate_password(self) -> None:
        """Проверяет минимальную длину пароля."""
        if len(self.password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
    def _hash_password(self, password: str) -> str:
        """Хэширует пароль пользователя."""
        # Генерация соли и хэширование
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def check_password(self, password: str) -> bool:
        """Проверяет, совпадает ли введенный пароль с хэшированным паролем."""
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

    def view_transactions(self) -> List['Transaction']:
        """Вывод всех транзакций."""
        return self.transactions.copy()

    def view_predictions(self) -> List['Prediction']:
        """Вывод всех предсказаний."""
        return self.predictions.copy()


@dataclass
class Administrator(User):

    """Класс для представления администратора. Наследуется из класса User"""

    def add_balance_to_user(self, user: User, amount: float) -> None:
        """Пополнение счета выбранного пользователя"""
        user.balance.add_balance(amount)
        
    def view_all_transactions(self, user: User) -> List['Transaction']:
        """Просмотр транзакций выбранного пользователя"""
        return user.view_transactions()
    
    def view_all_predictions(self, user: User) -> List['Prediction']:
        """Просмотр предсказаний модели для выбранного пользователя"""
        return user.view_predictions()
    

@dataclass
class Balance:

    """Класс для совершения операций с балансом пользователя
    
    Атрибуты:
        balance (float): остаток на счете
        amount (float): сумма пополнения или списания со счета    
    """

    balance: float = 0.0

    def add_balance(self, amount: float) -> None:
        """Пополнение баланса пользователем"""
        self.balance += amount
                
    def deduct_balance(self, amount: float) -> None:
        """Списание с баланса пользователя"""
        if self.balance >= amount:
            self.balance -= amount
        else:
            raise ValueError("Insufficient funds. Top up your balance and try again")
        
    def get_balance(self) -> float:
        """Вывод баланса пользователя."""
        return self.balance


@dataclass
class Model:
    """Класс для представления модели c проверкой входных данных.
    
    Атрибуты:
        param_1 (int): параметр 1
        param_2 (str): категориальная переменная 
        param_3 (float): параметр 3
        model_path (str): путь к модели
        input_data (Dict): словарь с входными данными
                    
    """
    param_1: int
    param_2: str
    param_3: float
    model_path: str
    input_data: Dict
    
    def __post_init__(self):
        """Инициализация класса."""
        self.model = self.load_model()

    def load_model(self):
        """Загрузка готовой модели из файла."""
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)
        
    def validate_param_1(self, param_1: int, min_val: int, max_val: int) -> int:
        """Проверка валидности первого параметра."""
        if param_1 is None or not (min_val <= param_1 <= max_val):
            raise ValueError(f"Enter value between {min_val} and {max_val}.")
        return param_1

    def validate_param_2(self, param_2: str) -> str:
        """Проверка валидности второго параметра."""
        categories = ['cat_1', 'cat_2', 'cat_3']
        if param_2 not in categories:
            raise ValueError(f"Param_2 must be one of {categories}.")
        return param_2
    
    def validate_param_3(self, param_3: float, min_val: float, max_val: float) -> float:
        """Проверка валидности третьего параметра."""
        if param_3 is None or not (min_val <= param_3 <= max_val):
            raise ValueError(f"Enter value between {min_val} and {max_val}.")
        return param_3
       
    def predict(self, input_data: Dict) -> float:
        """Расчет результата на основании параметров."""
        
        p1 = self.validate_param_1(input_data.get('param_1'), 20, 100)
        p2 = self.validate_param_2(input_data.get('param_2'))
        p3 = self.validate_param_3(input_data.get('param_3'), 0.0, 10.0)

        # Encode categorical variable
        cat_encoded = {'cat_1': 0, 'cat_2': 1, 'cat_3': 2}[p2]
        features = [[p1, cat_encoded, p3]]
        return self.model.predict(features)[0]
 

@dataclass
class Task(ABC):

    """Абстрактный базовый класс для представления операций с
       реализацией полиморфизма

    Атрибуты:
        task_id (int): идентификатор операции
        timestamp (datetime): метка времени совершения операции
    """

    task_id: int
    timestamp: datetime = field(default_factory=datetime.now)

    @abstractmethod
    def execute(self):
        pass


@dataclass
class History(ABC):

    """Абстрактный базовый класс для представления истории операций 
       конкретного пользователя с реализацией полиморфизма

    Атрибуты:
        user_id (int): идентификатор пользователя, которому принадлежат все операции
        timestamp (datetime): метка времени совершения операции
    """

    user_id: int
    timestamp: datetime = field(default_factory=datetime.now)

    @abstractmethod
    def to_dict(self):
        pass


@dataclass
class Prediction:
    
    """Класс фиксации результата выполнения операции по предсказанию.
       Говорит нам когда, кто выполнил операцию и какой результат получил.
       
    
    Атрибуты:
        user_id (int): идентификатор пользователя, сделавшего запрос
        task_id (int): идентификатор задачи
        result (float): результат работы модели
        cost (float): оплата услуги
        timestamp (datetime): метка времени выполнения операции
    
    """
    user_id: int
    task_id: int
    result: float
    cost: float
    timestamp: datetime=field(default_factory=datetime.now)


@dataclass
class PredictionTask(Task):
    
    """Класс для выполнения операции по предсказанию.
    """

    user: User
    model: Model
    input_data: Dict
    cost: float = 10.0

    def execute(self) -> Prediction:
        # Списание со счета пользователя средств
        self.user.balance.deduct_balance(amount=self.cost)

        result = self.model.predict(self.input_data)

        prediction = Prediction(
            user_id=self.user.id,
            task_id=self.task_id,
            result=result,
            cost=self.cost,
            timestamp=self.timestamp
        )
        # пополнение истории
        self.user.predictions.append(prediction)
        return prediction
    

@dataclass
class PredictionHistory(History):

    """Класс для представления записи для истории операций по предсказаниям.
    """

    task_id: int
    result: float
    cost: float

    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'task_id': self.task_id,
            'result': self.result,
            'cost': self.cost,
            'timestamp': self.timestamp
        }


@dataclass
class Transaction:

    """Класс для фиксации факта совершения транзакции."""
    
    user_id: int
    amount: float
    transaction_type: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'amount': self.amount,
            'transaction_type': self.transaction_type,
            'description': self.description,
            'timestamp': self.timestamp
        }

@dataclass
class TransactionTask(Task):

    """Класс для выполнения операций со счетом пользователя."""

    user: User
    amount: float
    transaction_type: str
    description: str

    def execute(self) -> Transaction:
        if self.transaction_type == "deposit":
            self.user.balance.add_balance(amount=self.amount)
        elif self.transaction_type == "withdraw":
            self.user.balance.deduct(amount=self.amount)
        else:
            raise ValueError("Transaction type must be 'deposit' or 'withdraw'.")

        transaction = Transaction(
            user_id=self.user.id,
            amount=self.amount,
            transaction_type=self.transaction_type,
            description=self.description,
            timestamp=self.timestamp
        )

        self.user.transactions.append(transaction)
        return transaction
    

@dataclass
class TransactionHistory(History):

    """Класс для представления истории транзакций."""

    amount: float
    transaction_type: str
    description: str

    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'amount': self.amount,
            'transaction_type': self.transaction_type,
            'description': self.description,
            'timestamp': self.timestamp
        }
