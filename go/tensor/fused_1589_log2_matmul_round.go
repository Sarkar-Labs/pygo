package tensor

import "fmt"

func Fused1589Log2MatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Round()
	return r, nil
}
func Fused1589Log2MatmulRoundName() string { return "log2_matmul_round" }
