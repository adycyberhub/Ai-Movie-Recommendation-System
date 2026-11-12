# src/main.py
from recommender import get_recommendations

def main():
    print("=" * 40)
    print("   AI MOVIE RECOMMENDATION SYSTEM   ")
    print("=" * 40)
    
    while True:
        movie_title = input("\nEnter a movie name (type 'exit' to quit): ").strip()
        
        if movie_title.lower() == 'exit':
            print("Exiting program. Goodbye!")
            break
            
        if not movie_title:
            print("[Error] Movie title cannot be empty. Please try again.")
            continue
            
        try:
            recommendations = get_recommendations(movie_title)
            if recommendations:
                print(f"\nTop 5 similar movies to '{movie_title}':")
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec}")
            else:
                print(f"[Notice] No recommendations found for '{movie_title}'.")
                
        except ValueError as e:
            print(f"[Error] {e}")
        except Exception as e:
            print(f"[Error] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()