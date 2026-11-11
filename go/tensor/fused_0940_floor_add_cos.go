package tensor

import "fmt"

func Fused0940FloorAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Cos()
	return r, nil
}
func Fused0940FloorAddCosName() string { return "floor_add_cos" }
