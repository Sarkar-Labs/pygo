package tensor

import "fmt"

func Fused1414SiluMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Sign()
	return r, nil
}
func Fused1414SiluMatmulSignName() string { return "silu_matmul_sign" }
