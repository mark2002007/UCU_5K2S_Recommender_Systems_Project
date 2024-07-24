from src.models.item_item_collaborative_filtering import ItemItemCollaborativeFiltering
from src.models.user_user_collaborative_filtering import UserUserColaborativeFiltering
from src.models.pagerank import PageRankRecommender
from src.models.popularity_based_recommender import PopularityBasedRecommender
from src.models.content import ContentBasedFiltering
from src.models.factorization_svd import SVDCollaborativeFiltering
from src.models.factorization_svd_als import SVDALSCollaborativeFiltering
from src.models.factorization_svd_grad import SVDGradientDescentRecommender
from src.models.factorization_svd_funk import FunkSVDCollaborativeFiltering
from src.models.neural_collaborative_filtering import NeuralCollaborativeFiltering
from src.models.multi_armed_bandits import MultiArmedBanditsRecommender
from src.models.ltr_linear_regression import LTRLinearRegressionRecommender