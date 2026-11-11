package tensor

import "fmt"

func Fused0228SigmoidDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Floor()
	return r, nil
}
func Fused0228SigmoidDivFloorName() string { return "sigmoid_div_floor" }
