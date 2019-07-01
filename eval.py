'''
可以修改script里面的各种参数：
    return {
        'IOU_CONSTRAINT': 0.5,
        'AREA_PRECISION_CONSTRAINT': 0.5,
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': 'res__([0-9]+).txt',
        'LTRB': False,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
        'CRLF': False,  # Lines are delimited by Windows CRLF format
        'CONFIDENCES': True,  # Detections must include confidence value. AP will be calculated
        'PER_SAMPLE_RESULTS': True  # Generate per sample results and produce data for visualization
    }
'''

from cal_recall.script import cal_recall_precison_f1

if __name__ == '__main__':
    gt_path = './gt/'
    result_path = 'C:/Users/AlienWare/Desktop/ja/Mask_RCNN_test/Mask_RCNN-master/samples/balloon/img12/'
    result = cal_recall_precison_f1(gt_path=gt_path, result_path=result_path)
    print(result)
