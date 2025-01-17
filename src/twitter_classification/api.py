from fastapi import FastAPI

# Create a FastAPI app instance
app = FastAPI()

# Define a root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# doesn't work currently:(