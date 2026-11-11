package tensor

import "fmt"

func Fused0923FloorDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_sqrt: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Sqrt()
	return r, nil
}
func Fused0923FloorDivSqrtName() string { return "floor_div_sqrt" }
