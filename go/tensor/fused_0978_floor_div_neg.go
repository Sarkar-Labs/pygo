package tensor

import "fmt"

func Fused0978FloorDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Neg()
	return r, nil
}
func Fused0978FloorDivNegName() string { return "floor_div_neg" }
