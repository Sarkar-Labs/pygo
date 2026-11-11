package tensor

import "fmt"

func Fused0938FloorDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Sin()
	return r, nil
}
func Fused0938FloorDivSinName() string { return "floor_div_sin" }
