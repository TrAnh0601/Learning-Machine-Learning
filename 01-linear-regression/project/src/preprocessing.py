from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
import numpy as np


def get_processor():
    """
    Create a preprocessing pipeline optimized for the House Price dataset.

    Key improvements:
    1. Domain-aware missing value handling
    2. Ordinal encoding for quality/condition features
    3. Skewness correction for numerical features
    4. Feature engineering
    """

    # Features where NA means "None" or "Not Present"
    # These are features where missing value is actually meaningful information
    categorical_na_as_none = [
        'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
    ]

    # Ordinal features with quality/condition ratings
    # These have inherent order: Excellent > Good > Average > Fair > Poor
    ordinal_features = {
        'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'GarageQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'GarageCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'PoolQC': ['None', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
        'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        'GarageFinish': ['None', 'Unf', 'RFn', 'Fin'],
        'Fence': ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']
    }

    # Numerical features that are typically very skewed
    skewed_features = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
        'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
    ]

    # Regular numerical features (not ordinal, not skewed)
    regular_numerical = [
        'MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt',
        'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold'
    ]

    # Nominal categorical features (no inherent order)
    nominal_categorical = [
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
        'MasVnrType', 'Foundation', 'Heating', 'CentralAir',
        'Electrical', 'Functional', 'GarageType', 'PavedDrive',
        'MiscFeature', 'SaleType', 'SaleCondition'
    ]

    # Function to handle skewness with log transformation
    def log_transform_skewed(X):
        # Add 1 to avoid log(0), similar to log1p
        return np.log1p(X)

    # Pipeline for skewed numerical features
    skewed_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(log_transform_skewed, validate=False)),
        ('scaler', StandardScaler())
    ])

    # Pipeline for regular numerical features
    regular_num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for ordinal features
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('ordinal', OrdinalEncoder(
            categories=list(ordinal_features.values()),
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    # Pipeline for nominal categorical features
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('skewed', skewed_transformer, skewed_features),
            ('regular_num', regular_num_transformer, regular_numerical),
            ('ordinal', ordinal_transformer, list(ordinal_features.keys())),
            ('nominal', nominal_transformer, nominal_categorical)
        ],
        remainder='drop'  # Drop any columns not explicitly listed
    )

    return preprocessor