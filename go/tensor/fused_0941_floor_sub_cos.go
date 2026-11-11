package tensor

import "fmt"

func Fused0941FloorSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Cos()
	return r, nil
}
func Fused0941FloorSubCosName() string { return "floor_sub_cos" }
