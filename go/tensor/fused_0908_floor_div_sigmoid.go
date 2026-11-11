package tensor

import "fmt"

func Fused0908FloorDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Sigmoid()
	return r, nil
}
func Fused0908FloorDivSigmoidName() string { return "floor_div_sigmoid" }
