export DRAKEMONGOURL='mongodb+srv://hunter:HunterC0dez!@drakecluster.wj44a.mongodb.net/myFirstDatabase?retryWrites=true&w=majority'
git clone https://github.com/wmcnicho/DrakeGenerator.git
cd DrakeGenerator
git checkout sacredSetup
pip3 install -r requirements.txt
python3 bars_custom.py with cell_type=gru num_epochs=1000 batch_size=256