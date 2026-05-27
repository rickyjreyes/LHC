## Download the LHCb request-48 input data

The input ROOT files for this analysis are available from the LHCb Open Data Ntupling Service.

Recommended local folder:

```text
data/lhcb_request_48/
```

## Option 1: HTTPS download

Create the data folder:

```bash
mkdir -p data/lhcb_request_48
cd data/lhcb_request_48
```

Download with `wget`:

```bash
wget https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/.filelist.yaml
wget https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/00382466_00000001_1.dvntuple.root
wget https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/00382466_00000002_1.dvntuple.root
wget https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/00382466_00000003_1.dvntuple.root
wget https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/00382467_00000001_1.dvntuple.root
wget https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/00382467_00000002_1.dvntuple.root
wget https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/00382467_00000003_1.dvntuple.root
```

Or download with `curl`:

```bash
curl -O https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/.filelist.yaml
curl -O https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/00382466_00000001_1.dvntuple.root
curl -O https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/00382466_00000002_1.dvntuple.root
curl -O https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/00382466_00000003_1.dvntuple.root
curl -O https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/00382467_00000001_1.dvntuple.root
curl -O https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/00382467_00000002_1.dvntuple.root
curl -O https://opendata-lhcb-ntupling-service.app.cern.ch/api/requests/48/outputs/real-production/00382467_00000003_1.dvntuple.root
```

## Option 2: XRootD download

Install XRootD:

```bash
conda install -c conda-forge xrootd
```

Verify installation:

```bash
xrdcp --version
```

Create the data folder:

```bash
mkdir -p data/lhcb_request_48
cd data/lhcb_request_48
```

Download with `xrdcp`:

```bash
xrdcp root://eospublic.cern.ch//eos/opendata/lhcb/upload/opendata-lhcb-ntupling-service/analysis-productions/merge-requests/5826/outputs/real-production/.filelist.yaml .
xrdcp root://eospublic.cern.ch//eos/opendata/lhcb/upload/opendata-lhcb-ntupling-service/analysis-productions/merge-requests/5826/outputs/real-production/00382466_00000001_1.dvntuple.root .
xrdcp root://eospublic.cern.ch//eos/opendata/lhcb/upload/opendata-lhcb-ntupling-service/analysis-productions/merge-requests/5826/outputs/real-production/00382466_00000002_1.dvntuple.root .
xrdcp root://eospublic.cern.ch//eos/opendata/lhcb/upload/opendata-lhcb-ntupling-service/analysis-productions/merge-requests/5826/outputs/real-production/00382466_00000003_1.dvntuple.root .
xrdcp root://eospublic.cern.ch//eos/opendata/lhcb/upload/opendata-lhcb-ntupling-service/analysis-productions/merge-requests/5826/outputs/real-production/00382467_00000001_1.dvntuple.root .
xrdcp root://eospublic.cern.ch//eos/opendata/lhcb/upload/opendata-lhcb-ntupling-service/analysis-productions/merge-requests/5826/outputs/real-production/00382467_00000002_1.dvntuple.root .
xrdcp root://eospublic.cern.ch//eos/opendata/lhcb/upload/opendata-lhcb-ntupling-service/analysis-productions/merge-requests/5826/outputs/real-production/00382467_00000003_1.dvntuple.root .
```

## Git note

The downloaded `.root` files are large binary data files and should not be committed to git.

Recommended `.gitignore` entry:

```gitignore
*.root
*.tmp
*.part
```