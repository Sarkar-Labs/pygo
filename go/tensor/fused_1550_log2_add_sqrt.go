package tensor

import "fmt"

func Fused1550Log2AddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Sqrt()
	return r, nil
}
func Fused1550Log2AddSqrtName() string { return "log2_add_sqrt" }
