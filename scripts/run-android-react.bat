@echo off
cd ../packages/frontend
call npm run build
call npx cap sync android
call npx cap open android
