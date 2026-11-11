package tensor

import "fmt"

func Fused1080RoundAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.ReLU()
	return r, nil
}
func Fused1080RoundAddReluName() string { return "round_add_relu" }
