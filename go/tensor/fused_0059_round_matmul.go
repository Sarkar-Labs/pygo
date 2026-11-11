package tensor

import "fmt"

func Fused0059RoundMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	return r, nil
}
func Fused0059RoundMatmulName() string { return "round_matmul" }
