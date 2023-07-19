audio_root="/home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/data/raw_val"
rm -f dnsmos_all_predictions_personalized.csv.old
rm -f dnsmos_all_predictions.csv.old
if [ -f "dnsmos_all_predictions_personalized.csv" ]; then
  mv dnsmos_all_predictions_personalized.csv dnsmos_all_predictions_personalized.csv.old
fi
if [ -f "dnsmos_all_predictions.csv" ]; then
  mv dnsmos_all_predictions.csv dnsmos_all_predictions.csv.old
fi
python3 dnsmos_local.py -t "$audio_root" -o dnsmos_all_predictions_personalized.csv -p
python3 dnsmos_local.py -t "$audio_root" -o dnsmos_all_predictions.csv