import numpy as np

def CCL(img):
    img = (img > 0).astype(int)
    rows, cols = img.shape
    
    label_img = np.zeros((rows, cols), dtype=int)
    next_label = 1
    
    max_labels = (rows * cols) // 2 + 1
    linked = np.zeros(max_labels, dtype=int)
    
    for r in range(rows):
        for c in range(cols):
            if img[r, c] == 1:
                neighbors = []
                
                if c > 0 and label_img[r, c-1] > 0:
                    neighbors.append(label_img[r, c-1])
                if r > 0 and label_img[r-1, c] > 0:
                    neighbors.append(label_img[r-1, c])
                    
                if not neighbors:
                    linked[next_label] = next_label
                    label_img[r, c] = next_label
                    next_label += 1
                else:
                    min_label = min(neighbors)
                    label_img[r, c] = min_label
                    
                    for n in neighbors:
                        root_n = n
                        while linked[root_n] != root_n:
                            root_n = linked[root_n]
                            
                        root_min = min_label
                        while linked[root_min] != root_min:
                            root_min = linked[root_min]
                            
                        if root_n != root_min:
                            if root_n < root_min:
                                linked[root_min] = root_n
                            else:
                                linked[root_n] = root_min

    total_labels = next_label - 1
    
    for i in range(1, total_labels + 1):
        root = i
        while linked[root] != root:
            root = linked[root]
        linked[i] = root
        
    unique_roots = np.unique(linked[1:total_labels + 1])
    num = len(unique_roots)
    label_map = np.zeros(total_labels + 1, dtype=int)
    
    for i, root in enumerate(unique_roots, start=1):
        label_map[root] = i
        
    for r in range(rows):
        for c in range(cols):
            if label_img[r, c] > 0:
                label_img[r, c] = label_map[linked[label_img[r, c]]]
                
    return label_img, num