@echo off
cd /d "C:\Users\Asus\Desktop\LandingPage\Chatboot"
call .venv\Scripts\activate
pip uninstall transformers -y
pip install transformers==4.40.0 --force-reinstall
pip install torch==2.0.0 --force-reinstall
pip install accelerate==0.30.0 --force-reinstall
echo "Installation complete"
pause