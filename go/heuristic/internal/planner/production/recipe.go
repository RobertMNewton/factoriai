package production

import (
	"fmt"
	"log"

	"github.com/factorio-ai/internal/data"
)

type Production struct {
	ProductID ProductID
	Quantity  float64
}

type Recipe struct {
	Input  []Production
	Output []Production

	Reference interface{}
	Category  string
}
type RecipeID int

type RecipeManager struct {
	catalog []Recipe
}

func (pm *ProductManager) NewProductionFrom(product interface{}, quantity float64) Production {
	productID, err := pm.GetProductID(product)
	if err != nil {
		log.Fatalf("Error: %e", err)
	}

	return Production{
		ProductID: productID,
		Quantity:  quantity,
	}
}

func NewRecipe(ref interface{}, input []Production, Output []Production) Recipe {
	if input == nil {
		input = make([]Production, 0, 4)
	}
	if Output == nil {
		Output = make([]Production, 0, 4)
	}

	return Recipe{
		Input:  input,
		Output: Output,

		Reference: ref,
	}
}

func (r Recipe) GetRef() interface{} {
	return r.Reference
}

func (r *Recipe) AddInput(product ProductID, quantity float64) {
	r.Input = append(r.Input, Production{
		ProductID: product,
		Quantity:  quantity,
	})
}

func (r *Recipe) AddOutput(product ProductID, quantity float64) {
	r.Input = append(r.Input, Production{
		ProductID: product,
		Quantity:  quantity,
	})
}

func (r Recipe) GetInput() []Production {
	return r.Input
}

func (r Recipe) GetOutput() []Production {
	return r.Output
}

func (r Recipe) GetInputFor(productID ProductID) (float64, bool) {
	// returns false if productID not found
	for _, production := range r.Input {
		if production.ProductID == productID {
			return production.Quantity, true
		}
	}
	return 0, false
}

func (r Recipe) GetOutputFor(productID ProductID) (float64, bool) {
	// returns false if productID not found
	for _, production := range r.Output {
		if production.ProductID == productID {
			return production.Quantity, true
		}
	}
	return 0, false
}

func (r Recipe) HasInput(productID ProductID) bool {
	for _, production := range r.Input {
		if production.ProductID == productID {
			return true
		}
	}
	return false
}

func (r Recipe) HasOutput(productID ProductID) bool {
	for _, production := range r.Output {
		if production.ProductID == productID {
			return true
		}
	}
	return false
}

func NewRecipeManager(size int) RecipeManager {
	return RecipeManager{
		catalog: make([]Recipe, 0, size),
	}
}

func NewRecipeManagerFrom(d data.Schema, pm *ProductManager) (rm RecipeManager) {
	rm.catalog = make([]Recipe, 0, len(d.Recipes)+len(d.Resources))
	for _, recipe_data := range d.Recipes {
		var recipe Recipe

		recipe.Input = make([]Production, 0, len(recipe_data.Ingredients))
		for _, ingredient := range recipe_data.Ingredients {
			productID, err := pm.GetProductID(ingredient.Name)
			if err != nil {
				log.Fatalf("Error: %e", err)
			}

			var production Production
			production.ProductID = productID
			production.Quantity = ingredient.Amount / recipe_data.EnergyRequired

			recipe.Input = append(recipe.Input, production)
		}

		recipe.Output = make([]Production, 0, len(recipe_data.Results))
		for _, result := range recipe_data.Results {
			productID, err := pm.GetProductID(result.Name)
			if err != nil {
				log.Fatalf("Error: %e", err)
			}

			var production Production
			production.ProductID = productID
			production.Quantity = result.Amount / recipe_data.EnergyRequired

			recipe.Output = append(recipe.Output, production)
		}

		recipe.Reference = recipe_data.Name
		recipe.Category = recipe_data.Category

		rm.catalog = append(rm.catalog, recipe)
	}

	for key, resource_data := range d.Resources {
		var recipe Recipe

		recipe.Output = make([]Production, 0, len(resource_data.Mineable.Results))
		for _, result := range resource_data.Mineable.Results {
			productID, err := pm.GetProductID(result.Name)
			if err != nil {
				log.Fatalf("Error: %e", err)
			}

			var production Production
			production.ProductID = productID
			production.Quantity = result.Amount / resource_data.Mineable.MiningTime

			recipe.Output = append(recipe.Output, production)
		}

		recipe.Reference = key
		recipe.Category = resource_data.Category

		rm.AddRecipe(recipe)
	}

	return rm
}

func (rm RecipeManager) Size() int {
	return len(rm.catalog)
}

func (rm *RecipeManager) AddRecipe(recipe Recipe) {
	rm.catalog = append(rm.catalog, recipe)
}

func (rm *RecipeManager) GetRecipe(recipeID RecipeID) Recipe {
	return rm.catalog[int(recipeID)]
}

func (rm *RecipeManager) GetRecipeID(ref interface{}) (RecipeID, error) {
	for id, recipe := range rm.catalog {
		if recipe.Reference == ref {
			return RecipeID(id), nil
		}
	}

	return RecipeID(-1), fmt.Errorf("no recipe found for %v", ref)
}

func (rm *RecipeManager) GetRecipes() []Recipe {
	return rm.catalog
}

func (rm *RecipeManager) GetCatalogProducing(productID ProductID) (res []struct {
	RecipeID
	float64
}) {
	// Finds the catalog producing the given product and returns a slice of uname structs containing the Recipe
	// and the amount of the given product it produces
	for recipeID, recipe := range rm.catalog {
		if quantity, ok := recipe.GetOutputFor(productID); ok {
			res = append(res, struct {
				RecipeID
				float64
			}{RecipeID(recipeID), quantity})
		}
	}

	return res
}

func (rm *RecipeManager) GetCatalogConsuming(productID ProductID) (res []struct {
	RecipeID
	float64
}) {
	// Finds the catalog producing the given product and returns a slice of uname structs containing the Recipe
	// and the amount of the given product it produces
	for recipeID, recipe := range rm.catalog {
		if quantity, ok := recipe.GetInputFor(productID); ok {
			res = append(res, struct {
				RecipeID
				float64
			}{RecipeID(recipeID), quantity})
		}
	}

	return res
}
