import pandas as pd
import config as cfg
from tqdm import tqdm
import config as cfg
from utils import *
from image_predictor import ImagePredictor
from clf_predictor import ClassifierPredictor
import warnings




def main():
    # Ignore specific warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.lib.function_base")
    print("data loading...")
    df_timediff = load_excel_sheet("data", sheet_name="120秒內解除轉轉樂紀錄")
    print("data has been successfully loaded")

    print("data preprocessing 0%")
    df_validation = df_timediff[["ChaNum", "ChaName", "UserID"]].drop_duplicates().reset_index(drop=True)
    df_validation = generate_playInSusMapRate(df_validation, df_timediff, cfg.sus_map_index)
    print("data preprocessing 10%.")
    df_validation = generate_maxTimesInOneDay(df_validation, df_timediff)
    print("data preprocessing 20%..")
    df_validation = generate_maxRepeatedPattern(df_validation, df_timediff)
    print("data preprocessing 30%...")
    df_validation = generate_maxNonRepeatedPattern(df_validation, df_timediff)
    print("data preprocessing 40%....")
    df_validation = generate_timediffAutocorr(df_validation, df_timediff)
    print("data preprocessing 50%.....")
    df_validation = generate_timediffMean(df_validation, df_timediff)
    print("data preprocessing 60%......")
    df_validation = generate_timediffStd(df_validation, df_timediff)
    print("data preprocessing 70%.......")
    df_validation = generate_idxmaxTimediffCount(df_validation, df_timediff)
    print("data preprocessing 80%........")
    df_validation = generate_maxTimediffCount(df_validation, df_timediff)
    print("data preprocessing 90%.........")
    df_validation = generate_timediffCountdiff(df_validation, df_timediff)
    print("data preprocessing 100%.........")

    for UserID in tqdm(df_timediff['UserID'].unique(), desc='Processing test-player-images', unit='player'):
        df = df_timediff[(df_timediff['UserID'] == UserID)]
        saveTimediffPlot(df, UserID, "test-player-images/0")

    print("finished preparing the images for CNN classification model...")
    print("inferencing CNN model...")
    current_dir = os.getcwd()
    test_dir = os.path.join(current_dir, 'test-player-images')
    cnn_model_path = "models/cnn-model-v2.h5"

    image_predictor = ImagePredictor(cnn_model_path, test_dir)
    image_predictor.preprocess_images()
    image_predictor.predict()
    image_predictor.create_dataframe()
    # print(image_predictor.df_cnn)

    # assume you have a DataFrame df_validation
    clf_model_path = "models/clf-model-v2.h5"

    print("inferencing CLF model...")
    clf_predictor = ClassifierPredictor(clf_model_path, df_validation)
    clf_predictor.load_model()
    clf_predictor.predict()
    # print(clf_predictor.df_validation)

    print("filtering the list and creating a cheater list...")
    df_clf_pred = clf_predictor.df_validation[clf_predictor.df_validation['Predicted_Probability']>cfg.CLF_prob_threshold]['UserID']
    df_cnn_pred = image_predictor.df_cnn[image_predictor.df_cnn['Predicted_Probability']>cfg.CNN_prob_threshold]['UserID']

    # cnn_clf_union = np.union1d(df_clf_pred, df_cnn_pred)
    # cnn_clf_union = pd.DataFrame(cnn_clf_union, columns=["UserID"])
    # cnn_clf_union.to_csv('output/cheater-list.csv', index=False)

    cnn_clf_intersection = pd.merge(df_clf_pred, df_cnn_pred, on='UserID')

    condition_1 = (df_validation['idxmaxTimediffCount'].isin(cfg.idmaxTimediffCount)) & (df_validation['timediffCountdiff'] >= 20)
    cheater_list_1 = df_validation.loc[condition_1, 'UserID'].tolist()

    condition_2 = (df_validation['idxmaxTimediffCount'].isin(cfg.idmaxTimediffCount)) & (df_validation['playInSusMapRate']>0.6) & (df_validation['timediffCountdiff']>=15)
    cheater_list_2 = df_validation.loc[condition_2, 'UserID'].tolist()

    condition_3 = (df_validation['playInSusMapRate']>0.9) & (df_validation['timediffCountdiff']>15)
    cheater_list_3 = df_validation.loc[condition_3, 'UserID'].tolist()

    df_combined = cnn_clf_intersection
    union_userid = np.union1d(df_combined['UserID'], cheater_list_1)
    union_userid = np.union1d(union_userid, cheater_list_2)
    union_userid = np.union1d(union_userid, cheater_list_3)
    df_combined = pd.DataFrame({'UserID': union_userid})
    df_combined["UserID"] = df_combined["UserID"].astype(str)
    df_combined = generate_iplist(df_combined, df_timediff)
    print("generate examption list...")
    df_combined = generate_exemptionlist(df_combined, cfg.userID_examption, cfg.ip_examption)

    print("-----------------------------")
    print("-----------------------------")
    total_cheaters = len(df_combined)
    exempted_cheaters = df_combined['Exemption'].sum()
    print(f"Total number of cheaters caught: {total_cheaters}")
    print(f"Number of cheaters exempted: {exempted_cheaters}")
    print("-----------------------------")
    print("-----------------------------")
    # df_combined.to_csv('/app/data/cheater-list.csv', index=False)
    print("final cheater list has been saved to your folder")
    df_combined.to_excel('output/cheater-list.xlsx', index=False)


if __name__ == '__main__':
    main()