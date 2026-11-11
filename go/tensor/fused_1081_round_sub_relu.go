package tensor

import "fmt"

func Fused1081RoundSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.ReLU()
	return r, nil
}
func Fused1081RoundSubReluName() string { return "round_sub_relu" }
