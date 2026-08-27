"""
Synthetic Financial Data Generator for FinSight.

Key Architecture Principles:
- Deterministic Balance: Includes an explicit opening balance transaction (+25000.00)
  so that SUM(transaction.amount) is completely deterministic.
- Money Sign Convention:
  - Inflows (salary, refund, opening balance) are POSITIVE (+).
  - Outflows (rent, groceries, bills, food) are NEGATIVE (-).
- Accessibility First: Generates default accessibility preferences for the demo user.
"""

import sys
import os
from datetime import datetime, date, timedelta
from decimal import Decimal

# Ensure workspace root is on sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.db import SessionLocal, init_db
from backend.models import User, Account, Transaction, Goal, Bill, Document


def seed_synthetic_data() -> None:
    """Populates SQLite database with rich synthetic financial data."""
    init_db()
    db = SessionLocal()

    try:
        # Check if already seeded
        existing_user = db.query(User).filter_by(email="aarav.sharma@example.com").first()
        if existing_user:
            print("Database already contains seed data for aarav.sharma@example.com.")
            return

        print("Seeding synthetic data for FinSight...")

        # 1. Create Demo User
        user = User(
            full_name="Aarav Sharma",
            email="aarav.sharma@example.com",
            accessibility_prefs={
                "voice_first": True,
                "screen_reader": True,
                "spoken_confirmations": True,
                "preferred_language": "en-IN",
            },
            is_active=True,
        )
        db.add(user)
        db.flush()

        # 2. Create Accounts
        primary_account = Account(
            user_id=user.id,
            name="HDFC Primary Savings",
            account_type="savings",
            balance=Decimal("24500.00"),  # Cached display balance
            monthly_income=Decimal("75000.00"),
            currency="INR",
            is_active=True,
        )
        db.add(primary_account)
        db.flush()

        # 3. Create Transactions with Money Sign Convention
        # Starting with an explicit Opening Balance transaction
        now = datetime.utcnow()
        start_of_month = now.replace(day=1, hour=9, minute=0, second=0, microsecond=0)

        transactions = [
            # Opening Balance (+25000.00)
            Transaction(
                account_id=primary_account.id,
                user_id=user.id,
                amount=Decimal("25000.00"),
                currency="INR",
                transaction_type="income",
                category="Other",
                description="Opening Balance",
                transaction_date=start_of_month - timedelta(days=30),
            ),
            # Monthly Salary (+75000.00)
            Transaction(
                account_id=primary_account.id,
                user_id=user.id,
                amount=Decimal("75000.00"),
                currency="INR",
                transaction_type="income",
                category="Other",
                merchant_name="TechCorp India Pvt Ltd",
                description="Monthly Salary Credit",
                transaction_date=start_of_month,
            ),
            # Rent Payment (-25000.00)
            Transaction(
                account_id=primary_account.id,
                user_id=user.id,
                amount=Decimal("-25000.00"),
                currency="INR",
                transaction_type="expense",
                category="Bills",
                merchant_name="Landlord Realty",
                description="Apartment Rent Payment",
                transaction_date=start_of_month + timedelta(days=1),
            ),
            # Groceries (-4200.00)
            Transaction(
                account_id=primary_account.id,
                user_id=user.id,
                amount=Decimal("-4200.00"),
                currency="INR",
                transaction_type="expense",
                category="Food",
                merchant_name="BigBasket",
                description="Monthly Groceries Order",
                transaction_date=start_of_month + timedelta(days=3),
            ),
            # Swiggy Food Delivery (-620.00)
            Transaction(
                account_id=primary_account.id,
                user_id=user.id,
                amount=Decimal("-620.00"),
                currency="INR",
                transaction_type="expense",
                category="Food",
                merchant_name="Swiggy",
                description="Dinner Delivery",
                transaction_date=start_of_month + timedelta(days=5),
            ),
            # Transport / Metro (-800.00)
            Transaction(
                account_id=primary_account.id,
                user_id=user.id,
                amount=Decimal("-800.00"),
                currency="INR",
                transaction_type="expense",
                category="Transport",
                merchant_name="Namma Metro",
                description="Metro Card Recharge",
                transaction_date=start_of_month + timedelta(days=7),
            ),
            # Electricity Bill (-1850.00)
            Transaction(
                account_id=primary_account.id,
                user_id=user.id,
                amount=Decimal("-1850.00"),
                currency="INR",
                transaction_type="expense",
                category="Bills",
                merchant_name="BESCOM",
                description="Electricity Utility Payment",
                transaction_date=start_of_month + timedelta(days=10),
            ),
            # Shopping (-3030.00)
            Transaction(
                account_id=primary_account.id,
                user_id=user.id,
                amount=Decimal("-3030.00"),
                currency="INR",
                transaction_type="expense",
                category="Shopping",
                merchant_name="Amazon India",
                description="Audio Headset and Ergonomic Keyboard",
                transaction_date=start_of_month + timedelta(days=12),
            ),
        ]
        db.add_all(transactions)

        # 4. Create Goals
        goals = [
            Goal(
                user_id=user.id,
                name="Emergency Rainy Day Fund",
                target_amount=Decimal("150000.00"),
                current_amount=Decimal("45000.00"),
                monthly_contribution=Decimal("10000.00"),
                currency="INR",
                target_date=date(2027, 3, 31),
                status="active",
            ),
            Goal(
                user_id=user.id,
                name="Smart Braille Display",
                target_amount=Decimal("50000.00"),
                current_amount=Decimal("20000.00"),
                monthly_contribution=Decimal("5000.00"),
                currency="INR",
                target_date=date(2026, 12, 31),
                status="active",
            ),
        ]
        db.add_all(goals)

        # 5. Create Bills
        bills = [
            Bill(
                user_id=user.id,
                name="BESCOM Electricity",
                amount=Decimal("1850.00"),
                currency="INR",
                category="Bills",
                due_date=date(now.year, now.month, 5) if now.day < 5 else (now + timedelta(days=20)).date(),
                frequency="monthly",
                status="unpaid",
                is_recurring=True,
            ),
            Bill(
                user_id=user.id,
                name="Airtel Fiber Broadband",
                amount=Decimal("1179.00"),
                currency="INR",
                category="Bills",
                due_date=(now + timedelta(days=10)).date(),
                frequency="monthly",
                status="unpaid",
                is_recurring=True,
            ),
            Bill(
                user_id=user.id,
                name="Apartment Maintenance",
                amount=Decimal("3500.00"),
                currency="INR",
                category="Bills",
                due_date=(now + timedelta(days=15)).date(),
                frequency="monthly",
                status="unpaid",
                is_recurring=True,
            ),
        ]
        db.add_all(bills)

        # 6. Create Sample Document metadata
        doc = Document(
            user_id=user.id,
            filename="bescom_electricity_bill.pdf",
            file_path="/storage/documents/bescom_electricity_bill.pdf",
            document_type="bill",
            mime_type="application/pdf",
            raw_text="BESCOM Electricity Bill Account: 1048291 Amount: Rs 1,850.00 Due Date: 2026-09-05",
            extracted_facts={
                "vendor": "BESCOM",
                "amount": 1850.00,
                "due_date": "2026-09-05",
                "account_number": "1048291",
            },
            is_suspicious=False,
        )
        db.add(doc)

        db.commit()
        print("Synthetic financial data seeded successfully.")

    except Exception as e:
        db.rollback()
        print(f"Error seeding synthetic data: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_synthetic_data()
