import numpy as np

np.random.seed(42)

array = np.random.randint(1, 51, size=(5, 4))

print("Generated 2D Array (5x4):")
print(array)
print()

rows, cols = array.shape

anti_diagonal_elements = []
for i in range(min(rows, cols)):
    anti_diagonal_elements.append(array[i, cols-1-i])

print("Anti-diagonal elements (Method 1 - List comprehension):")
print(anti_diagonal_elements)
print()

flipped_array = np.fliplr(array)
anti_diagonal_np = np.diagonal(flipped_array)

print("Anti-diagonal elements (Method 2 - NumPy fliplr + diagonal):")
print(anti_diagonal_np)
print()

n_elements = min(rows, cols)
row_indices = np.arange(n_elements)
col_indices = np.arange(cols-1, cols-1-n_elements, -1)
anti_diagonal_fancy = array[row_indices, col_indices]

print("Anti-diagonal elements (Method 3 - Fancy indexing):")
print(anti_diagonal_fancy)
print()

print("Anti-diagonal positions and values:")
for i in range(min(rows, cols)):
    row_idx = i
    col_idx = cols - 1 - i
    print(f"Position ({row_idx}, {col_idx}): {array[row_idx, col_idx]}")

print("\n" + "="*50)
print("MAXIMUM VALUE IN EACH ROW")
print("="*50)


row_maxima = np.max(array, axis=1)

print("Maximum value in each row:")
print(row_maxima)
print()

print("Detailed breakdown:")
for i in range(rows):
    max_val = np.max(array[i])
    max_pos = np.argmax(array[i])
    print(f"Row {i}: {array[i]} -> Max = {max_val} (at column {max_pos})")
print()

overall_mean = np.mean(array)
print(f"Overall mean of the array: {overall_mean:.2f}")
print()


mask = array <= overall_mean
print("Boolean mask (True for elements <= mean):")
print(mask)
print()


elements_le_mean = array[mask]
print("Elements less than or equal to the mean:")
print(elements_le_mean)
print()

count_le_mean = np.sum(mask)
total_elements = array.size
percentage = (count_le_mean / total_elements) * 100

print(f"Count of elements <= mean: {count_le_mean} out of {total_elements}")
print(f"Percentage: {percentage:.1f}%")
print()


print("Positions and values of elements <= mean:")
row_indices, col_indices = np.where(mask)
for i in range(len(row_indices)):
    row_idx = row_indices[i]
    col_idx = col_indices[i]
    value = array[row_idx, col_idx]
    print(f"Position ({row_idx}, {col_idx}): {value}")
print()

filtered_array = elements_le_mean
print("New array containing only elements <= mean:")
print(f"Shape: {filtered_array.shape}")
print(f"Array: {filtered_array}")
def numpy_boundary_traversal(matrix):
    """
    Traverse the boundary of a numpy matrix in clockwise order starting from top-left corner.
    
    Args:
        matrix (numpy.ndarray): 2D numpy array to traverse
        
    Returns:
        list: Elements visited along the boundary in clockwise order
        
    Raises:
        ValueError: If matrix is not 2D or is empty
    """

    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D array")
    
    if matrix.size == 0:
        raise ValueError("Input array cannot be empty")
    
    rows, cols = matrix.shape
    boundary_elements = []
    
    if rows == 1:
        return matrix[0].tolist()
    
    if cols == 1:
        
        return matrix[:, 0].tolist()
    
    
    
    for j in range(cols - 1):
        boundary_elements.append(matrix[0, j])
    
    
    for i in range(rows - 1):
        boundary_elements.append(matrix[i, cols - 1])
    
  
    for j in range(cols - 1, 0, -1):
        boundary_elements.append(matrix[rows - 1, j])
    
    
    for i in range(rows - 1, 0, -1):
        boundary_elements.append(matrix[i, 0])
    
    return boundary_elements



def test_boundary_traversal():
    """Test the boundary traversal function with different matrix sizes and types."""
    
    print("="*60)
    print("NUMPY BOUNDARY TRAVERSAL FUNCTION TESTS")
    print("="*60)
    
   
    print("\nTest 1: 3x3 Matrix")
    matrix1 = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
    print("Matrix:")
    print(matrix1)
    result1 = numpy_boundary_traversal(matrix1)
    print(f"Boundary traversal: {result1}")
    print("Expected: [1, 2, 3, 6, 9, 8, 7, 4]")
    
    
    print("\nTest 2: 4x5 Matrix")
    matrix2 = np.arange(1, 21).reshape(4, 5)
    print("Matrix:")
    print(matrix2)
    result2 = numpy_boundary_traversal(matrix2)
    print(f"Boundary traversal: {result2}")
    print("Expected: [1, 2, 3, 4, 5, 10, 15, 20, 19, 18, 17, 16, 11, 6]")
    
   
    print("\nTest 3: 2x2 Matrix")
    matrix3 = np.array([[1, 2],
                        [3, 4]])
    print("Matrix:")
    print(matrix3)
    result3 = numpy_boundary_traversal(matrix3)
    print(f"Boundary traversal: {result3}")
    print("Expected: [1, 2, 4, 3]")
    
    
    print("\nTest 4: Single Row (1x5)")
    matrix4 = np.array([[1, 2, 3, 4, 5]])
    print("Matrix:")
    print(matrix4)
    result4 = numpy_boundary_traversal(matrix4)
    print(f"Boundary traversal: {result4}")
    print("Expected: [1, 2, 3, 4, 5]")
   
    print("\nTest 5: Single Column (5x1)")
    matrix5 = np.array([[1], [2], [3], [4], [5]])
    print("Matrix:")
    print(matrix5)
    result5 = numpy_boundary_traversal(matrix5)
    print(f"Boundary traversal: {result5}")
    print("Expected: [1, 2, 3, 4, 5]")
    
   
    print("\nTest 6: 5x4 Matrix with Random Values")
    np.random.seed(42)
    matrix6 = np.random.randint(10, 100, size=(5, 4))
    print("Matrix:")
    print(matrix6)
    result6 = numpy_boundary_traversal(matrix6)
    print(f"Boundary traversal: {result6}")
    

    print("\nVisual representation of boundary traversal order:")
    print("Matrix with positions marked:")
    traversal_order = np.full((5, 4), '  ', dtype='U3')
    boundary = numpy_boundary_traversal(matrix6)
    
  
    rows, cols = matrix6.shape
    order = 1
    
    for j in range(cols - 1):
        traversal_order[0, j] = f'{order:2d}'
        order += 1
    
    for i in range(rows - 1):
        traversal_order[i, cols - 1] = f'{order:2d}'
        order += 1
    
    for j in range(cols - 1, 0, -1):
        traversal_order[rows - 1, j] = f'{order:2d}'
        order += 1
    
    for i in range(rows - 1, 0, -1):
        traversal_order[i, 0] = f'{order:2d}'
        order += 1
    
    print("Traversal order (numbers show sequence):")
    for row in traversal_order:
        print('  '.join(row))


if __name__ == "__main__":
    test_boundary_traversal()
    
    print("\n" + "="*60)
    print("APPLYING TO ORIGINAL ARRAY")
    print("="*60)
    
    np.random.seed(42)
    original_array = np.random.randint(1, 51, size=(5, 4))
    print("\nOriginal 5x4 array:")
    print(original_array)
    
    boundary_result = numpy_boundary_traversal(original_array)
    print(f"\nBoundary traversal result: {boundary_result}")

