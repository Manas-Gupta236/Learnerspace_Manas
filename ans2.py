import numpy as np

random_array = np.random.uniform(0, 10, 20)

print("Original array:")
print(random_array)

rounded_array = np.round(random_array, 2)

print("\nArray rounded to 2 decimal places:")
print(rounded_array)

min_value = np.min(rounded_array)
max_value = np.max(rounded_array)
median_value = np.median(rounded_array)

print(f"\nStatistics:")
print(f"Minimum: {min_value}")
print(f"Maximum: {max_value}")
print(f"Median: {median_value}")

modified_array = rounded_array.copy()
mask = modified_array < 5
modified_array[mask] = modified_array[mask] ** 2

print(f"\nArray after replacing elements < 5 with their squares:")
print(modified_array)

def numpy_alternate_sort(array):
    """
    Takes a 1D numpy array and returns a new array with elements sorted 
    in an alternating pattern: smallest, largest, second smallest, second largest, etc.
    
    Parameters:
    array (numpy.ndarray): Input 1D numpy array
    
    Returns:
    numpy.ndarray: New array with alternating sort pattern
    """
    sorted_array = np.sort(array)
    
    
    result = np.zeros_like(sorted_array)
    
    left = 0 
    right = len(sorted_array) - 1  
    
    for i in range(len(sorted_array)):
        if i % 2 == 0:  
            result[i] = sorted_array[left]
            left += 1
        else:
            result[i] = sorted_array[right]
            right -= 1
    
    return result

alternate_sorted = numpy_alternate_sort(modified_array)
print(f"\nAlternate sorted array (smallest, largest, 2nd smallest, 2nd largest, ...):")
print(alternate_sorted)