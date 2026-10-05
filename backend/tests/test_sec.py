from app.services.sec_client import SECClient


sec = SECClient()

data = sec.get_company_submissions("320193")

print("Company:")
print(data["name"])

print()

print("CIK:")
print(data["cik"])

print()

print("Tickers:")
print(data["tickers"])

print()

print("Exchanges:")
print(data["exchanges"])