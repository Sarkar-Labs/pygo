package tensor

import "fmt"

func Fused0903FloorDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.ReLU()
	return r, nil
}
func Fused0903FloorDivReluName() string { return "floor_div_relu" }
