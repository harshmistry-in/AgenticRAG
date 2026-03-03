if __name__ == "__main__":
    
    import uvicorn
    uvicorn.run("src.app:app", host="localhost", port=8001, reload=True)