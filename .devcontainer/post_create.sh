# Ensure that the .env file exists
if [ -f .env ]; then
  # Export all variables from the .env file
  export $(grep -v '^#' .env | xargs) 
  
  # Configure Git with the exported variables
  git config --global user.name "$GIT_USER_NAME"
  git config --global user.email "$GIT_USER_EMAIL"
else
  echo ".env file not found. Skipping Git configuration."
fi
