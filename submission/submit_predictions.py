def submit_predictions(prediction, outputFilename, sampleSubmissionFilename):
    '''
    Function to generate predictions to be used in Kaggle submission
    '''
    import csv
    
    ## Read the indices 
    with open(sampleSubmissionFilename,'r') as csvinput:
        reader = csvinput.read().splitlines()
        i=-1
        ind = []
        pred_rating = []
        for row in reader:
            if i != -1:
                pos, default_rating = row.split(',')
                row, col = pos.split("_")
                row = int(row.replace("r", ""))
                col = int(col.replace("c", ""))       
                pred_rating.append(prediction[row-1, col-1])
                ind.append(pos)
            i+=1
    ## Create rows to be written
    rows = zip(ind, pred_rating)

    ## Write prediction with indices
    import csv
    with open(outputFilename, 'w', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(['Id','Prediction'])
        for row in rows:
            writer.writerow(row)