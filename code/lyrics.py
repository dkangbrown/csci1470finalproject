# Deep learning model to generate lyrics

# Use the millionsongSubset dataset to generate lyrics
# Use the lyrics from the dataset to train the model
# Use the lyrics to generate new lyrics

# Import the necessary libraries
from utils import *

# Load the dataset
data = pd.read_csv('data/lyrics.csv')
data.head()

# Check for missing values
data.isnull().sum()

# Drop the missing values
data.dropna(inplace=True)

# Check for missing values
data.isnull().sum()

# Check the shape of the dataset
data.shape

# Check the first few rows of the dataset
data.head()

# Check the distribution of the genre column

