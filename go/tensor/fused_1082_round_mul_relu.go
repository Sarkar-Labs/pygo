package tensor

import "fmt"

func Fused1082RoundMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.ReLU()
	return r, nil
}
func Fused1082RoundMulReluName() string { return "round_mul_relu" }
