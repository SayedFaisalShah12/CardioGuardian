"""
Main Pipeline Script for CardioGuardian
Runs the complete ML pipeline: preprocessing, training, and evaluation
"""

from data_preprocessing import load_data, preprocess_data, save_scaler
from train_model import compare_models, select_best_model, save_model
from evaluate_model import evaluate_model, print_evaluation_report, plot_confusion_matrix, plot_roc_curve


def main():
    """Run the complete ML pipeline"""
    
    print("=" * 70)
    print("CardioGuardian - Complete ML Pipeline")
    print("=" * 70)
    
    # Step 1: Load and preprocess data
    print("\n[Step 1/4] Loading and preprocessing data...")
    print("-" * 70)
    df = load_data('heart.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    save_scaler(scaler)
    
    # Step 2: Compare and train models
    print("\n[Step 2/4] Training and comparing models...")
    print("-" * 70)
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # Step 3: Select and save best model
    print("\n[Step 3/4] Selecting and saving best model...")
    print("-" * 70)
    best_name, best_model, best_accuracy = select_best_model(results)
    save_model(best_model, best_name, 'models/best_model.pkl')
    
    # Step 4: Evaluate the best model
    print("\n[Step 4/4] Evaluating best model...")
    print("-" * 70)
    metrics = evaluate_model(best_model, X_test, y_test)
    print_evaluation_report(metrics)
    
    # Generate plots
    plot_confusion_matrix(metrics['confusion_matrix'])
    if metrics['y_pred_proba'] is not None:
        plot_roc_curve(y_test, metrics['y_pred_proba'])
    
    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)
    print(f"\nBest Model: {best_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nNext steps:")
    print("1. Run 'streamlit run app.py' to launch the web application")
    print("2. Use 'predict.py' for programmatic predictions")
    print("=" * 70)


if __name__ == "__main__":
    main()

