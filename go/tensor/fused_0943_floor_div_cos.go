package tensor

import "fmt"

func Fused0943FloorDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Cos()
	return r, nil
}
func Fused0943FloorDivCosName() string { return "floor_div_cos" }
