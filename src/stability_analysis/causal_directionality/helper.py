from src.stability_analysis.permute_grn.helper import main as main_permute


def main(par):
    par = {
        **par,
        'analysis_types': ['direction', 'weight'],
        'degrees': [100],
    }
    main_permute(par)
