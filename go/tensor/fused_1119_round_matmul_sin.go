package tensor

import "fmt"

func Fused1119RoundMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Sin()
	return r, nil
}
func Fused1119RoundMatmulSinName() string { return "round_matmul_sin" }
