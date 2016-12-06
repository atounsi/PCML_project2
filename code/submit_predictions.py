def submit_predictions(prediction, outputFilename, sampleSubmissionFilename):
    import csv
    
    ## Read the indices 
    with open('../data/sampleSubmission.csv','r') as csvinput:
        reader = csvinput.read().splitlines()
        i=-1
        ind = []
        for row in reader:
            if i != -1:
                pos, rating = row.split(',')
                ind.append(pos)
            i+=1
    ## Create rows to be written
    rows = zip(ind, prediction)

    ## Write prediction with indices
    import csv
    with open(outputFilename, 'w', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(['Id','Prediction'])
        for row in rows:
            writer.writerow(row)