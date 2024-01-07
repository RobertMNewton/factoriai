package production

import (
	"fmt"

	"github.com/factorio-ai/internal/data"
)

type ProductID int

type ProductManager struct {
	catalog []interface{}
}

func NewProductManager() ProductManager {
	return ProductManager{
		catalog: make([]interface{}, 0, 128),
	}
}

func NewProductManagerFrom(d data.Schema) (pm ProductManager) {
	pm.catalog = make([]interface{}, 0, len(d.Items))
	for _, item := range d.Items {
		pm.catalog = append(pm.catalog, item.Name)
	}

	return pm
}

func (pm ProductManager) Size() int {
	return len(pm.catalog)
}

func (pm *ProductManager) AddProduct(new_product ProductID) {
	pm.catalog = append(pm.catalog, new_product)
}

func (pm ProductManager) GetProduct(productID ProductID) (interface{}, error) {
	if int(productID) >= len(pm.catalog) || int(productID) < 0 {
		return nil, fmt.Errorf("got invalid productID: received %d, expected between %d and %d", productID, 0, len(pm.catalog))
	}
	return pm.catalog[productID], nil
}

func (pm ProductManager) GetProductID(product interface{}) (ProductID, error) {
	for i, v := range pm.catalog {
		if v == product {
			return ProductID(i), nil
		}
	}
	return ProductID(-1), fmt.Errorf("unable to find product %v in catalog", product)
}
