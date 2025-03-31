import os
import sys

if __name__ == "__main__":
    # Print diagnostic information
    print(f"Current working directory: {os.getcwd()}")
    print(f"Location of this file: {os.path.abspath(__file__)}")
    print(f"Config path should be: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cfg')}")
    
    # Run the main function
    from eureka.eureka import main
    main() 