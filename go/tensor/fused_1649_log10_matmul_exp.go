package tensor

import "fmt"

func Fused1649Log10MatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Exp()
	return r, nil
}
func Fused1649Log10MatmulExpName() string { return "log10_matmul_exp" }
