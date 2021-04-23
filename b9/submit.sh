# Local command:
export DRAKEMONGOURL='REDACTED'
git clone https://github.com/wmcnicho/DrakeGenerator.git
cd DrakeGenerator
git checkout sacredSetup
pip3 install -r requirements.txt
python3 bars_custom.py with cell_type=gru num_epochs=5