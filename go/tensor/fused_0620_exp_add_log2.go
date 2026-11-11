package tensor

import "fmt"

func Fused0620ExpAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Log2()
	return r, nil
}
func Fused0620ExpAddLog2Name() string { return "exp_add_log2" }
