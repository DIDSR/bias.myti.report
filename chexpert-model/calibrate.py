from args import TestArgParser
from constants import *
from saver import ModelSaver
from data import get_loader
from calibration_utils import *
from predict import Predictor

def calibrate_model(args):
    print("Beginning model calibration")
    # argument management
    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args
    ckpt_path = model_args.ckpt_path
    # check if a prediction file was given
    if args.prediction_file == "None":
        args.prediction_file = None
    if args.prediction_file is None:
        # load model
        ckpt_save_dir = Path(ckpt_path).parent
        model_uncertainty = model_args.model_uncertainty
        model_args, transform_args\
            = ModelSaver.get_args(cl_model_args=model_args,
                                dataset=data_args.dataset,
                                ckpt_save_dir=ckpt_save_dir,
                                model_uncertainty=model_uncertainty)
        model_args.moco = args.model_args.moco
        model, ckpt_info = ModelSaver.load_model(ckpt_path=ckpt_path,
                                                gpu_ids=args.gpu_ids,
                                                model_args=model_args,
                                                is_training=False)
        # get valid loader
        valid_loader = get_loader(phase=data_args.phase,
                                data_args=data_args,
                                transform_args=transform_args,
                                is_training=False,
                                return_info_dict=True,
                                logger=None)
        # get predictions 
        predictor = Predictor(model=model, device=args.device, code_dir=args.code_dir)
        predictions, groundtruth, paths = predictor.predict(valid_loader)
        
    else: # load predictions and groundtruth
        predictions = pd.read_csv(args.prediction_file)
        groundtruth = pd.read_csv(args.prediction_file.replace('predictions', 'groundtruth'))

    combined = groundtruth.copy()
    pred_cols = []
    label_cols = []
    for t in data_args.custom_tasks.split(","):
        combined[f'{t} gt'] = combined['Path'].map(predictions.set_index('Path')[t])
        pred_cols.append(f"{t} gt")
        label_cols.append(t)
    preds = combined[pred_cols].values
    labels = combined[label_cols].values
    # get prediction with subclass information [info_pred]
    orig_df = pd.read_csv(data_args.test_csv)
    # carry-over from older generate_paritions
    orig_df = orig_df.rename(columns={"Black_or_African_American":"Black"})
    # set which columns we want to pull from the original csv 
    info_cols = ['F','M','Black', "White", "Yes", 'No']
    info_pred = predictions.copy()
    cols = [c for c in info_pred.columns if c not in ['Path']]
    info_pred = info_pred.rename(columns={c:f"{c} score" for c in cols})
    for c in info_cols:
        info_pred[c] = info_pred['Path'].map(orig_df.set_index("Path")[c])
    # Platt scaling 
    # get before metrics
    b_metrics = calibration_metrics(predictions=preds.ravel(), labels=labels.ravel())
    b_metrics['Calibrated'] = 'No'
    # # fit and save logistic regression model
    PS = PlattScaling()
    PS.fit(preds, labels.ravel())
    PS.save(ckpt_path.replace('best.pth.tar', 'platt_scale.pkl'))
    CC_info = {'validation':{'labels':labels.ravel(), 'preds':preds}}
    # get after metrics - > save
    a_metrics = calibration_metrics(predictions=PS.scale(preds)[:,1], labels=labels.ravel())
    a_metrics['Calibrated'] = 'Yes'
    cal_metrics= pd.concat([b_metrics, a_metrics],ignore_index=True)
    metric_fp = ckpt_path.replace("best.pth.tar","validation_calibration_metrics_ps.json")
    cal_metrics.to_json(metric_fp, orient='table', indent=2)

    # load test predictions/labels (to look at cal curve) # TODO: remove (DEBUG)
    # test_predictions = pd.read_csv("/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/DB_ensemble_BETSY/attempt_1/decision_boundaries/decision_boundaries/DB_samples_RAND_0.csv", index_col=0)
    # test_gt = pd.read_csv("/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/DB_ensemble_BETSY/attempt_1/decision_boundaries/decision_boundaries/DB_samples.csv")
    # test_predictions['labels'] = test_predictions['Path'].map(test_gt.set_index("Path")['Yes'])
    # test_preds = test_predictions['Yes'].values.reshape(-1, 1)
    # CC_info['test'] = {'labels':test_predictions['labels'].values.ravel(), 'preds':test_preds}
    # PS.plot_curves(CC_info, ckpt_path.replace('best.pth.tar', 'calibration_curve.png'))
    # # view histograms of prediction dists before/after scaling for test/validation
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2,2,figsize=(8,6))
    # binwidth=0.025
    
    # ax[0,0].hist(preds, bins=np.arange(0,1,binwidth))
    # ax[0,0].set_title("Validation (Uncalibrated)")
    # ax[0,0].set_xlabel("Mean predicted probability")
    # ax[0,0].set_ylabel("Count")

    # ax[0,1].hist(PS.scale(preds)[:,1], bins=np.arange(0,1,binwidth))
    # ax[0,1].set_title("Validation (Calibrated)")
    # ax[0,1].set_xlabel("Mean predicted probability")
    # ax[0,1].set_ylabel("Count")

    # ax[1,0].hist(test_preds, bins=np.arange(0,1,binwidth))
    # ax[1,0].set_title("Test (Uncalibrated)")
    # ax[1,0].set_xlabel("Mean predicted probability")
    # ax[1,0].set_ylabel("Count")

    # ax[1,1].hist(PS.scale(test_preds)[:,1], bins=np.arange(0,1,binwidth))
    # ax[1,1].set_title("Test (Calibrated)")
    # ax[1,1].set_xlabel("Mean predicted probability")
    # ax[1,1].set_ylabel("Count")
    
    # plt.tight_layout()
    # plt.suptitle("Prediction Distributions")
    # plt.savefig(ckpt_path.replace('best.pth.tar', 'example_dist.png'), dpi=300)
    # plt.close()
    

if __name__ == "__main__":
    parser = TestArgParser()
    parser.parser.add_argument("--calibration_mode", choices=['temperature'])
    parser.parser.add_argument("--tasks", dest='data_args.custom_tasks') # not sure why this isn't importing from model properly
    parser.parser.add_argument("--prediction_file", default=None)
    calibrate_model(parser.parse_args())
    print()