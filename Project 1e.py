#src/evaluate.py
import torch
from torch.utils.data import DataLoader, random_split
from data import RFSignalDataset
from model import ScaledModulationResNet
def evaluate():
    torch.manual_seed(42)
    dataset = RFSignalDataset(num_samples_per_class=1500)
    _, test_set = random_split(dataset, [int(0.8* len(dataset)), len(dataset) - int(0.8* len(dataset))])
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    model = ScaledModulationResNet(num_classes=4)
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()
    correct, total = 0, 0
    snr_bins = [-5, 0, 5, 10, 15, 20]
    bin_correct = {b: 0 for b in snr_bins}
    bin_total = {b: 0 for b in snr_bins}
    with torch.no_grad():
        for inputs, labels, snrs in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for pred, label, snr in zip(preds, labels, snrs):
                total += 1
                if pred == label:
                    correct += 1
                closest_bin = min(snr_bins, key=lambda x: abs(x - snr.item()))
                bin_total[closest_bin] += 1
                if pred == label:
                    bin_correct[closest_bin] += 1
    print(f"\nOverall Test Accuracy: {100* correct / total:.2f}%")
    print("\n Accuracy Breakdown by SNR ")
    for b in snr_bins:
        if bin_total[b] > 0:
            acc = 100* bin_correct[b] / bin_total[b]
            print(f"SNR ~ {b:2d} dB: {acc:6.2f}% (Samples: {bin_total[b]})")
if __name__ == "__main__":
    evaluate()