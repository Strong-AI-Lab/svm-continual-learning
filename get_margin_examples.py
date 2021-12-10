def extract_anchors_ranking(self, y_pred, y_true):
	#keep indexes of anchor examples per class, for next iteration
	pos_indexes_list = [[]]  #a row is a class when that class is positive, each column is an instance index        
	
	#extract SVs and save them for replay in next tasks        
	for i in range(0, self.task_id + 1):
		pos_idx = i
							
		pos_y = y_pred[:, pos_idx]
		pos_y_true = y_true[:, pos_idx]
		
		#filter true only
		#pos_margin_idx_true = np.where(pos_y_true == 1)[0]
		
		#calculate distance to margin, arrange by that distance and select subset
		y_pred_distance_margin = abs(1 - pos_y) #pos_y
		y_pred_distance_margin = y_pred_distance_margin * pos_y_true #y_pred_distance_margin[np.isin(y_pred_distance_margin, pos_margin_idx_true)]
		
		#just keep positives, cause these are correct class + closer
		y_pred_distance_margin[y_pred_distance_margin < 0] = 100000 #only positives (actual class)
		
		y_pred_distance_margin_idx = np.argsort(y_pred_distance_margin) #indexes of sorted array

		#only keep_sv of current class, all the previous keep all (as SVs have been selected already previously!)
		if i == self.task_id:
			y_pred_distance_margin_idx = y_pred_distance_margin_idx[0:int(len(np.where(y_pred_distance_margin != 100000)[0]) * self.keep_sv)]
		else:
			y_pred_distance_margin_idx = np.where(y_pred_distance_margin != 100000)[0]
		
		anchors = y_pred_distance_margin_idx
		
		if len(anchors) > 0:            
			anchors = np.concatenate((np.repeat(i, len(anchors)).reshape(-1,1), 
									   anchors.reshape(-1,1)), axis = 1)
			
			if len(self.anchors) == 0:
				self.anchors = anchors
				self.anchors = self.anchors.astype(int) 
			else:
				self.anchors = np.append(self.anchors, anchors, axis = 0)
				self.anchors = self.anchors.astype(int)
				
				
def extract_anchors_ranking_posneg(self, y_pred, y_true):
	#keep indexes of anchor examples per class, for next iteration
	pos_indexes_list = [[]]  #a row is a class when that class is positive, each column is an instance index        
	
	#extract SVs and save them for replay in next tasks - classes 0 and 1 only         
	for i in range(0, self.task_id + 1):
		pos_idx = i
							
		pos_y = y_pred[:, pos_idx]
		pos_y_true = y_true[:, pos_idx]
		
		neg_y = y_pred[:, [x for x in range(0, self.out_dim) if x != i]]
		neg_y_true = y_true[:, [x for x in range(0, self.out_dim) if x != i]]
		
		
		#calculate distance to margin, arrange by that distance and select subset
		y_pred_distance_margin = abs(1 - pos_y) #pos_y
		y_pred_distance_margin = y_pred_distance_margin * pos_y_true 

		y_pred_distance_margin_neg = abs(1 - neg_y) #pos_y
		y_pred_distance_margin_neg = y_pred_distance_margin_neg * neg_y_true 
		
		#just keep positives, cause these are correct class + closer
		y_pred_distance_margin[y_pred_distance_margin < 0] = 100000 #only positives (actual class)
		y_pred_distance_margin_neg[y_pred_distance_margin_neg < 0] = 100000 
		
		y_pred_distance_margin_idx = np.argsort(y_pred_distance_margin) #indexes of sorted array
		y_pred_distance_margin_idx_neg = np.argsort(np.min(y_pred_distance_margin_neg, axis = 1))

		#only keep_sv of current class, all the previous keep all (as SVs have been selected already previously!)
		if i == self.task_id:
			y_pred_distance_margin_idx = y_pred_distance_margin_idx[0:int(len(np.where(y_pred_distance_margin != 100000)[0]) * self.keep_sv)]
			y_pred_distance_margin_idx_neg = y_pred_distance_margin_idx_neg[0:int(len(np.where(y_pred_distance_margin_neg != 100000)[0]) * self.keep_sv)]
		else:
			y_pred_distance_margin_idx = np.where(y_pred_distance_margin != 100000)[0]
			#negatives will still need to be filtered though
			y_pred_distance_margin_idx_neg = y_pred_distance_margin_idx_neg[0:int(len(np.where(y_pred_distance_margin_neg != 100000)[0]) * self.keep_sv)]
		
		anchors = np.concatenate((y_pred_distance_margin_idx, y_pred_distance_margin_idx_neg))
		
		if len(anchors) > 0:            
			anchors = np.concatenate((np.repeat(i, len(anchors)).reshape(-1,1), anchors.reshape(-1,1)), axis = 1)
			
			if len(self.anchors) == 0:
				self.anchors = anchors
				self.anchors = self.anchors.astype(int) 
			else:
				self.anchors = np.append(self.anchors, anchors, axis = 0)
				self.anchors = self.anchors.astype(int) 
